import gym
import numpy as np

def factory(env):

  env_type = type(env)

  class FilteredEnv(env_type):

    def __init__(self):
      # transfer properties
      self.__dict__.update(env.__dict__)

      # state
      self.state_mean = (env.observation_space.high + env.observation_space.low) / 2
      self.state_diff = (env.observation_space.high - env.observation_space.low) / 2

      # actions
      self.actions_mean = (env.action_space.high + env.action_space.low) / 2
      self.actions_diff = (env.action_space.high - env.action_space.low) / 2

      self.observation_space = gym.spaces.Box(self.filter_state(env.observation_space.low),
                                              self.filter_state(env.observation_space.high))
      self.action_space = gym.spaces.Box(- np.ones_like(env.action_space.high), np.ones_like(env.action_space.high))

    def step(self, action):
      filtered_action = np.clip(self.filter_action(action), self.action_space.low, self.action_space.high)

      state, reward, done, info = env_type.step(self, filtered_action)
      filtered_state = self.filter_state(state)

      return filtered_state, reward, done, info

    def filter_state(self, state):
      return (state - self.state_mean) / self.state_diff

    def filter_action(self, action):
      return action * self.actions_diff + self.actions_mean

  return FilteredEnv()