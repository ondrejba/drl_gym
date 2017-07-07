import gym
import matplotlib.pyplot as plt

from utils.Prep import Player

ALG_NAME = "deep-q-network"

env = gym.make("Breakout-v0")
env = gym.wrappers.Monitor(env, ALG_NAME, force=True)

player = Player(Player.Type.ATARI)

while True:
  obs, done = env.reset(), False
  episode_rew = 0

  while not done:
    env.render()
    state, reward, done, _ = env.step(env.action_space.sample())

    x = player.process(obs)

    plt.imshow(obs)
    plt.show()

    if not x[1]:
      plt.imshow(x[0])
      plt.show()

    episode_rew += reward

env.close()
