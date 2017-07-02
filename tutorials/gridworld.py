import random
from enum import Enum
import numpy as np

class Gridworld(object):

  class Actions(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

  def __init__(self, width, height, rewards):
    self.width = width
    self.height = height
    self.rewards = rewards

  def get_reward(self, x, y):
    return self.rewards[x, y]

  def get_possible_actions(self, x, y):
    actions = [self.Actions.LEFT, self.Actions.RIGHT, self.Actions.UP, self.Actions.DOWN]

    if y == 0:
      actions.remove(self.Actions.LEFT)
    if y == self.width - 1:
      actions.remove(self.Actions.RIGHT)
    if x == 0:
      actions.remove(self.Actions.UP)
    if x == self.height - 1:
      actions.remove(self.Actions.DOWN)

    return actions

class DPAgent(object):

  def __init__(self, world):
    self.world = world

    self.values = np.zeros((world.width, world.height))
    self.policy = np.zeros((world.width, world.height), dtype=Gridworld.Actions)

    for i in range(world.width):
      for j in range(world.height):
        actions = world.get_possible_actions(i, j)
        self.policy[i, j] = random.choice(actions)

    self.discount = 0.9

  def policy_evaluation(self, convergence=0.01):
    while True:
      delta = 0

      for i in range(world.width):
        for j in range(world.height):
          tmp = self.values[i, j]
          self.values[i, j] = world.get_reward(i, j) + self.discount * self.values[self.act(i, j, self.policy[i, j])]  
          delta = max(delta, abs(tmp - self.values[i, j]))

      if delta < convergence:
        break

  def policy_improvement(self):
    while True:
      policy_stable = True

      for i in range(world.width):
        for j in range(world.height):
          tmp = self.policy[i, j]
          actions = world.get_possible_actions(i, j)

          best_action = None
          highest_value = None

          for action in actions:
            value = self.world.get_reward(i, j) + self.discount * self.values[self.act(i, j, action)]

            if highest_value is None or value > highest_value:
              best_action = action
              highest_value = value

          self.policy[i, j] = best_action

          if tmp != self.policy[i, j]:
            policy_stable = False

      if policy_stable:
        break

  def act(self, x, y, direction):
    if direction == Gridworld.Actions.LEFT:
      y -= 1
    elif direction == Gridworld.Actions.RIGHT:
      y += 1
    elif direction == Gridworld.Actions.UP:
      x -= 1
    elif direction == Gridworld.Actions.DOWN:
      x += 1

    if y < 0 or y >= self.world.width:
      raise Exception("Out of bounds: x = %d." % x)
    if x < 0 or x >= self.world.height:
      raise Exception("Out of bounds: y = %d." % y)

    return x, y


rewards = np.zeros((5, 5))
rewards[4, 4] = 1
rewards[3, 3] = -1


world = Gridworld(5, 5, rewards)

agent = DPAgent(world)

for i in range(100):
  agent.policy_evaluation()
  agent.policy_improvement()

print(agent.values)
print(agent.policy)