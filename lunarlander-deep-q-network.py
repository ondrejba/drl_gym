import gym, argparse

from agents.DeepQNetwork import DeepQNetwork
import utils.utils as utils
from utils.architect import NeuralNetwork
import utils.config as config

ALG_NAME = "data/deep-q-network"

def main(args):
  env = gym.make("LunarLander-v2")

  if not args.disable_monitor:
    monitor_callable = utils.MonitorCallable(args.monitor_frequency)
    env = gym.wrappers.Monitor(env, ALG_NAME, force=True, video_callable=monitor_callable.call)

  nn_config = {
    "hidden": [128, 64, 32],
    "batch_norm": False
  }
  nn = NeuralNetwork(nn_config, NeuralNetwork.Type.MLP)
  agent = DeepQNetwork(nn, 9, 4, ALG_NAME, learning_rate=args.learning_rate, hard_update_frequency=args.hard_update_frequency,
                       buffer_size=args.buffer_size, soft_update_rate=args.soft_update_rate,
                       max_iters=args.num_steps, exploration=0.2, max_reward=200)

  total_score = 0

  ep_idx = 0
  while True:
    score, step = agent.run_episode(env)
    total_score += score

    if step > args.num_steps:
      break

    if ep_idx != 0 and ep_idx % 100 == 0:
      print("step %d, average score over 100 episodes: %f" % (step, total_score / 100))
      total_score = 0

    ep_idx += 1

  agent.close()

  env.close()

  if not args.disable_upload and not args.disable_monitor:
    gym.upload(ALG_NAME, api_key=config.API_KEY)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--learning-rate", type=float, default=1e-6)
  parser.add_argument("--hard-update-frequency", type=int, default=1000)
  parser.add_argument("--soft-update-rate", type=float, default=None)
  parser.add_argument("--buffer-size", type=int, default=250000)
  parser.add_argument("--num-steps", type=int, default=1000000)

  parser.add_argument("--disable-upload", action="store_true", default=False)
  parser.add_argument("--disable-monitor", action="store_true", default=False)

  parser.add_argument("--monitor-frequency", type=int, default=100)

  parsed = parser.parse_args()
  main(parsed)