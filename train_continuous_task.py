import argparse, gym, os
import tensorflow as tf

import utils.utils as utils
import utils.policy as policy
import utils.config as config
from utils.Prep import Prep
import utils.Simulation as Simulation

def main(args):
  # algorithm name and monitor directory
  alg_name = "{}_{}".format(args.env, args.agent)
  monitor_directory = os.path.join("data", alg_name)

  # setup the environment
  env = Simulation.factory(gym.make(args.env))

  if not args.disable_monitor:
    monitor_callable = utils.MonitorCallable(args.monitor_frequency)
    env = gym.wrappers.Monitor(env, monitor_directory, force=True, video_callable=monitor_callable.call)

  state_shape = env.observation_space.shape
  if len(state_shape) == 1:
    state_shape = state_shape[0]

  action_shape = env.action_space.shape
  if len(action_shape) == 1:
    action_shape = action_shape[0]

  # setup state preparation
  if args.prep.lower() == "simple":
    prep = Prep(Prep.Type.EXPAND_DIM)
  elif args.prep.lower() == "atari":
    prep = Prep(Prep.Type.ATARI)
  else:
    raise ValueError("Unknown prep. type.")

  # setup exploration policy
  if args.policy.lower() == "none":
    exp_policy = policy.ContinuousPolicy()
  elif args.policy.lower() == "ou":
    exp_policy = policy.OrnsteinUhlenbeckNoise(action_shape)
  elif args.policy.lower() == "ou_anneal":
    exp_policy = policy.OrnsteinUhlenbeckNoiseAnneal(action_shape, 0.5, args.num_steps, 0.1)
  elif args.policy.lower() == "gaussian_anneal":
    exp_policy = policy.GaussianNoiseAnneal(policy.GaussianNoiseAnneal.Mode.LINEAR, action_shape, 0.5, args.num_steps,
                                            0.1)
  else:
    raise ValueError("Unknown exploration policy.")

  # set random seeds
  tf.set_random_seed(2018)
  env.seed(2018)

  # setup agent
  if args.agent.lower() == "naf":
    from agents.NAF import NAF
    args.build = NAF.Build[args.naf_build.upper()]
    agent = NAF(prep, args.build, exp_policy, state_shape, action_shape, monitor_directory,
                num_steps=args.num_steps, learning_rate=args.learning_rate, update_rate=args.update_rate,
                batch_size=args.batch_size, buffer_size=args.buffer_size, max_reward=args.max_reward,
                train_freq=args.train_freq, steps_before_train=args.steps_before_train)
  elif args.agent.lower() == "ddpg":
    from agents.DDPG import DDPG
    agent = DDPG(prep, exp_policy, state_shape, action_shape, env.action_space.high, env.action_space.low, monitor_directory, num_steps=args.num_steps,
                 buffer_size=args.buffer_size, max_reward=args.max_reward, steps_before_train=args.steps_before_train,
                 detail_summary=args.detail_summary, batch_size=args.batch_size)

  else:
    raise ValueError("Unknown agent.")

  # learn
  ep_idx = 0

  while True:

    if ep_idx % args.evaluation_frequency == 0 and ep_idx != 0:
      total_score = 0

      for eval_idx in range(args.num_evaluations):
        score, _ = agent.run_episode(env, eval=True, ep_idx=ep_idx)
        total_score += score
        ep_idx += 1

      avg_score = total_score / args.num_evaluations
      print("{:d}: avg. score over {:d} evaluations: {:.02f}".format(ep_idx, args.num_evaluations, avg_score))

    _, step = agent.run_episode(env, eval=False, ep_idx=ep_idx)

    if step > args.num_steps:
      break

    ep_idx += 1

  agent.close()
  env.close()

  if not args.disable_upload and not args.disable_monitor:
    gym.upload(monitor_directory, api_key=config.API_KEY)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("env", help="name of the environment to play")
  parser.add_argument("agent", help="name of the agent to use")
  parser.add_argument("prep", help="simple, atari")
  parser.add_argument("policy", help="none, ou, ou_anneal, gaussian_anneal")

  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--update-rate", type=float, default=1e-3)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--buffer-size", type=int, default=1000000)
  parser.add_argument("--train-freq", type=int, default=1)
  parser.add_argument("--steps-before-train", type=int, default=10000)

  parser.add_argument("--naf-build", default="single")

  parser.add_argument("--num-steps", type=int, default=100000)
  parser.add_argument("--max-reward", type=float, default=None)

  parser.add_argument("--disable-upload", action="store_true", default=False)
  parser.add_argument("--disable-monitor", action="store_true", default=False)

  parser.add_argument("--detailed-summary", action="store_true", default=False)
  parser.add_argument("--monitor-frequency", type=int, default=100, help="0 to disable monitor")
  parser.add_argument("--log-frequency", type=int, default=100)
  parser.add_argument("--evaluation-frequency", type=int, default=100)
  parser.add_argument("--num-evaluations", type=int, default=5)
  parser.add_argument("--detail-summary", action="store_true")

  parsed = parser.parse_args()
  main(parsed)