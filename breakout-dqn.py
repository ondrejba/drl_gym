import gym, argparse

from agents.DeepQNetwork import DeepQNetwork
from utils.Prep import Prep
from utils.architect import NeuralNetwork
import utils.utils as utils
import utils.config as config

ALG_NAME = "data/deep-q-network"

def main(args):
  env = gym.make("Breakout-v0")

  if not args.disable_monitor:
    monitor_callable = utils.MonitorCallable(args.monitor_frequency)
    env = gym.wrappers.Monitor(env, ALG_NAME, force=True, video_callable=monitor_callable.call)

  prep = Prep(Prep.Type.ATARI)

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
  dqn = DeepQNetwork(nn, prep, (Prep.ATARI_WIDTH, Prep.ATARI_HEIGHT, 4), 4, ALG_NAME,
                     buffer_size=args.buffer_size, hard_update_frequency=args.hard_update_frequency,
                     soft_update_rate=args.soft_update_rate, max_iters=args.num_steps,
                     lin_exp_end_iter=0.3, lin_exp_final_eps=0.1, steps_before_learn=args.steps_before_learn,
                     train_freq=args.train_freq)

  total_score = 0

  ep_idx = 0
  while True:
    score, step = dqn.run_episode(env)
    total_score += score

    if step > args.num_steps:
      break

    if ep_idx != 0 and ep_idx % 100 == 0:
      print("step %d, average score over 100 episodes: %f" % (step, total_score / 100))
      total_score = 0

    ep_idx += 1

  dqn.close()
  env.close()

  if not args.disable_upload and not args.disable_monitor:
    gym.upload(ALG_NAME, api_key=config.API_KEY)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--hard-update-frequency", type=int, default=1000)
  parser.add_argument("--soft-update-rate", type=float, default=None)
  parser.add_argument("--buffer-size", type=int, default=10000)
  parser.add_argument("--num-steps", type=int, default=10000000)
  parser.add_argument("--steps-before-learn", type=int, default=10000)
  parser.add_argument("--train-freq", type=int, default=4)

  parser.add_argument("--disable-upload", action="store_true", default=False)
  parser.add_argument("--disable-monitor", action="store_true", default=False)

  parser.add_argument("--monitor-frequency", type=int, default=100, help="0 to disable monitor")

  parsed = parser.parse_args()
  main(parsed)