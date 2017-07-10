import argparse, gym, os
import tensorflow as tf

from agents.DeepQNetwork import DeepQNetwork
from agents.MonteCarloPolicyGradient import MonteCarloPolicyGradient
from utils.Prep import Prep
from utils.architect import NeuralNetwork
import utils.utils as utils
import utils.config as config
import utils.policy as policy

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
    exp_policy = policy.Greedy()
  elif args.policy.lower() == "e-greedy":
    exp_policy = policy.EpsilonGreedy(0.2)
  elif args.policy.lower() == "e-greedy-anneal":
    exp_policy = policy.EpsilonGreedyAnneal(policy.EpsilonGreedyAnneal.Mode.LINEAR,
                                            0.5, args.num_steps, 0.1)
  else:
    raise ValueError("Unknown exploration policy.")

  # set random seeds
  tf.set_random_seed(2018)
  env.seed(2018)

  # setup network builder
  nn = NeuralNetwork({
    "conv": [
      {
        "num_maps": 32,
        "filter_shape": (8, 8),
        "stride": (4, 4)
      },
      {
        "num_maps": 64,
        "filter_shape": (4, 4),
        "stride": (2, 2)
      },
      {
        "num_maps": 64,
        "filter_shape": (3, 3),
        "stride": (1, 1)
      }
    ],
    "pool": None,
    "hidden": [
      512
    ]
  }, NeuralNetwork.Type.CNN_MLP)

  # setup agent
  if args.agent.lower() == "dqn":
    agent = DeepQNetwork(nn, prep, exp_policy, state_shape, action_shape, monitor_directory,
                       buffer_size=args.buffer_size, hard_update_frequency=args.hard_update_freq,
                       soft_update_rate=args.soft_update_rate, num_steps=args.num_steps,
                       steps_before_learn=args.steps_before_learn, train_freq=args.train_freq)
  elif args.agent.lower() == "mcpg":
    agent = MonteCarloPolicyGradient(state_shape, action_shape, monitor_directory)
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
  parser.add_argument("policy", help="none, e-greedy, e-greedy-anneal")

  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--soft-update-rate", type=float, default=None)
  parser.add_argument("--hard-update-freq", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--buffer-size", type=int, default=100000)
  parser.add_argument("--train-freq", type=int, default=1)

  parser.add_argument("--num-steps", type=int, default=100000)
  parser.add_argument("--steps-before-learn", type=int, default=1000)
  parser.add_argument("--max-reward", type=int, default=None)

  parser.add_argument("--disable-upload", action="store_true", default=False)
  parser.add_argument("--disable-monitor", action="store_true", default=False)

  parser.add_argument("--detailed-summary", action="store_true", default=False)
  parser.add_argument("--monitor-frequency", type=int, default=100, help="0 to disable monitor")
  parser.add_argument("--log-frequency", type=int, default=100)

  parsed = parser.parse_args()
  main(parsed)