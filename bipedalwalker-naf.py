import gym, argparse
import utils.config as config

from agents.NAF import NAF
from utils.Prep import Prep
import utils.utils as utils
import utils.policy as policy

ALG_NAME = "data/NAF"

def main(args):
  env = gym.make("BipedalWalker-v2")

  if not args.disable_monitor:
    monitor_callable = utils.MonitorCallable(args.monitor_frequency)
    env = gym.wrappers.Monitor(env, ALG_NAME, force=True, video_callable=monitor_callable.call)

  prep = Prep(Prep.Type.EXPAND_DIM)

  if args.build.lower() == "single":
    build = NAF.Build.SINGLE
  elif args.build.lower() == "multiple":
    build = NAF.Build.MULTIPLE
  elif args.build.lower() == "hydra":
    build = NAF.Build.HYDRA
  else:
    raise ValueError("Wrong build type.")

  agent_policy = policy.GaussianNoiseAnneal(policy.GaussianNoiseAnneal.Mode.LINEAR, 4, 0.5, args.num_steps, 0.1)

  agent = NAF(prep, build, agent_policy, 24, 4, ALG_NAME, learning_rate=args.learning_rate, update_rate=args.update_rate,
              buffer_size=args.buffer_size, max_iters=args.num_steps, detailed_summary=args.detailed_summary)

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

  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--update-rate", type=float, default=1e-3)
  parser.add_argument("--buffer-size", type=int, default=100000)
  parser.add_argument("--num-steps", type=int, default=500000)
  parser.add_argument("--build", default="single")

  parser.add_argument("--disable-upload", action="store_true", default=False)
  parser.add_argument("--disable-monitor", action="store_true", default=False)

  parser.add_argument("--detailed-summary", action="store_true", default=False)
  parser.add_argument("--monitor-frequency", type=int, default=100, help="0 to disable monitor")

  parsed = parser.parse_args()
  main(parsed)