import gym

ALG_NAME = "data/deep-q-network"
API_KEY = "sk_rQdUjIzcR1eUcA7YwgrVSw"

num_steps = 500000

env = gym.make("LunarLander-v2")
env = gym.wrappers.Monitor(env, ALG_NAME, force=True)

from agents.DeepQNetwork import DeepQNetwork

agent = DeepQNetwork(8, 4, ALG_NAME, layers=[64, 32, 4], learning_rate=1e-3, update_frequency=2000, buffer_size=50000,
                     max_iters=num_steps, lin_exp_end_iter=0.5, lin_exp_final_eps=0.02, max_reward=200)

total_score = 0

ep_idx = 0
while True:
  score, step = agent.run_episode(env)
  total_score += score

  if step > num_steps:
    break

  if ep_idx != 0 and ep_idx % 100 == 0:
    print("step %d, average score over 100 episodes: %f" % (step, total_score / 100))
    total_score = 0

  ep_idx += 1

agent.close()

env.close()
gym.upload(ALG_NAME, api_key=API_KEY)

