import random
import numpy as np
from enum import Enum

"""
DISCRETE
"""

class DiscretePolicy:

  def select_action(self, values):
    pass

  def reset(self):
    pass

class Greedy(DiscretePolicy):

  def select_action(self, values):
    return np.argmax(values)

class EpsilonGreedy(DiscretePolicy):

  def __init__(self, epsilon):
    self.epsilon = epsilon

  def select_action(self, values):
    rand = random.random()

    if rand <= self.epsilon:
      return random.randint(0, len(values) - 1)
    else:
      return np.argmax(values)

class EpsilonGreedyAnneal(DiscretePolicy):

  class Mode(Enum):
    LINEAR = 1

  def __init__(self, mode, fract_iters, max_iters, final_epsilon):

    self.mode = mode
    self.fract_iters = fract_iters
    self.max_iters = max_iters
    self.final_epsilon = final_epsilon

    self.step = 0

  def select_action(self, values):
    action =  max(self.final_epsilon, 1 - self.step / (self.fract_iters * self.max_iters))
    self.step += 1
    return action

"""
CONTINUOUS
"""

class ContinuousPolicy:

  def add_noise(self, action):
    return action

  def reset(self):
    pass

class OrnsteinUhlenbeckNoise(ContinuousPolicy):
  # Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py

  def __init__(self, action_dim, sigma=0.1, mu=0, theta=0.05):
    self.action_dim = action_dim

    self.mu = mu
    self.theta = theta
    self.sigma = sigma

    self.state = np.ones(self.action_dim) * self.mu
    self.reset()

  def add_noise(self, action):
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
    self.state = x + dx

    return action + self.state

  def reset(self):
    self.state = np.ones(self.action_dim) * self.mu

class OrnsteinUhlenbeckNoiseAnneal(ContinuousPolicy):

  class Mode(Enum):
    LINEAR = 1

  def __init__(self, action_dim, fract_iters, max_iters, final_fract, sigma=0.3, theta=0.15, mu=0):
    self.action_dim = action_dim

    self.mu = mu
    self.theta = theta
    self.sigma = sigma

    self.fract_iters = fract_iters
    self.max_iters = max_iters
    self.final_fract = final_fract

    self.state = np.ones(self.action_dim) * self.mu
    self.step = 0
    self.reset()

  def add_noise(self, action):
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
    self.state = x + dx

    action = action + self.state * max(self.final_fract, (self.step / (self.fract_iters * self.max_iters)))
    self.step += 1
    return action

  def reset(self):
    self.state = np.ones(self.action_dim) * self.mu

class GaussianNoiseAnneal(ContinuousPolicy):

  class Mode(Enum):
    LINEAR = 1

  def __init__(self, mode, action_dim, fract_iters, max_iters, final_fract):
    self.action_dim = action_dim
    self.mode = mode

    self.fract_iters = fract_iters
    self.max_iters = max_iters
    self.final_fract = final_fract

    self.step = 0

  def add_noise(self, action):
    action = action + np.random.randn(self.action_dim) * max(self.final_fract, (self.step / (self.fract_iters * self.max_iters)))
    self.step += 1
    return action

  def reset(self):
    pass