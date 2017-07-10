import argparse, gym, os

import utils.utils as utils
import utils.policy as policy
import utils.config as config
from utils.Prep import Prep
from agents.NAF import NAF

def main(args):
  # algorithm name and monitor directory
  alg_name = "{}_{}".format(args.env, args.agent)
  monitor_directory = os.path.join("data", alg_name)

  # setup the environment
  env = gym.make(args.env)

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

  # setup agent
  if args.agent.lower() == "naf":
    agent = NAF(prep, NAF.Build.SINGLE, exp_policy, state_shape, action_shape, monitor_directory,
                num_steps=args.num_steps)
  else:
    raise ValueError("Unknown agent.")

  # learn
  total_score = 0
  ep_idx = 0

  while True:
    score, step = agent.run_episode(env)
    total_score += score

    if step > args.num_steps:
      break

    if ep_idx != 0 and ep_idx % args.log_frequency == 0:
      print("step %d, average score over %d episodes: %.2f" % (step, args.log_frequency, total_score / 100))
      total_score = 0

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

  parser.add_argument("--num-steps", type=int, default=100000)

  parser.add_argument("--disable-upload", action="store_true", default=False)
  parser.add_argument("--disable-monitor", action="store_true", default=False)

  parser.add_argument("--detailed-summary", action="store_true", default=False)
  parser.add_argument("--monitor-frequency", type=int, default=100, help="0 to disable monitor")
  parser.add_argument("--log-frequency", type=int, default=100)

  parsed = parser.parse_args()
  main(parsed)