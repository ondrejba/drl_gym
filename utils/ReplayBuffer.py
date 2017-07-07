import numpy as np

class ReplayBuffer():

  def __init__(self, size, state_dim, action_dim):
    self.size = size
    self.next_idx = 0
    self.full = False

    self.states = np.empty((self.size, *state_dim))
    self.actions = np.empty((self.size, action_dim))
    self.rewards = np.empty(self.size)
    self.next_states = np.empty((self.size, *state_dim))
    self.done = np.empty(self.size)

  def add(self, item):

    self.states[self.next_idx] = item["state"]
    self.actions[self.next_idx] = item["action"]
    self.rewards[self.next_idx] = item["reward"]
    self.next_states[self.next_idx] = item["next_state"]
    self.done[self.next_idx] = item["done"]

    if self.next_idx == self.size - 1:
      self.full = True

    self.next_idx = (self.next_idx + 1) % self.size

  def sample(self, size):

    if not self.full:
      idxs = np.random.randint(0, self.next_idx, size=size)
    else:
      idxs = np.random.randint(0, self.size, size=size)

    return {
      "states": self.states[idxs, :],
      "actions": self.actions[idxs, :],
      "rewards": self.rewards[idxs],
      "next_states": self.next_states[idxs, :],
      "done": self.done[idxs]
    }

  def compute_state_mean_and_std(self):
    if self.full:
      end = self.size
    else:
      end = self.next_idx

    state_mean = np.mean(self.states[:end], axis=0)
    state_std = np.std(self.states[:end], axis=0)

    return state_mean, state_std

  def normalize_states(self, mean, std):
    if self.full:
      end = self.size
    else:
      end = self.next_idx

    self.states[:end] = (self.states[:end] - mean) / std
    self.next_states[:end] = (self.next_states[:end] - mean) / std

  def get_size(self):
    if not self.full:
      return self.next_idx
    else:
      return self.size
