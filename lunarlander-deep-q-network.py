import gym

ALG_NAME = "data/deep-q-network"
API_KEY = "sk_rQdUjIzcR1eUcA7YwgrVSw"

num_episodes = 50000

env = gym.make("LunarLander-v2")
env = gym.wrappers.Monitor(env, ALG_NAME, force=True)

from agents.DeepQNetwork import DeepQNetwork

agent = DeepQNetwork(8, 4, ALG_NAME)

moving_avg = 0
alpha = 0.001

for i in range(num_episodes):
  score = agent.run_episode(env, i)

  moving_avg += alpha * (score - moving_avg)

  if i != 0 and i % 100 == 0:
    print("%d, moving average return: %f" % (i, moving_avg))

agent.close()

env.close()
gym.upload(ALG_NAME, api_key=API_KEY)
