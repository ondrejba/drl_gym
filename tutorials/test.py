# sk_rQdUjIzcR1eUcA7YwgrVSw

import gym, math
import numpy as np

#env = gym.make('FrozenLake-v0')

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

values = np.zeros((4, 4))
policy = np.zeros((4, 4))
holes = [(2, 2), (2, 4), (3, 4), (4, 1)]
start = (1, 1)
finish = (4, 4)

discount = 0.9
threshold = 0.01

"""
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("observation: %d" % observation)
        env.render()

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
"""

def policy_eval():
  
  while True:
    delta = 0

    for i in range(4):
      for j in range(4):
        tmp = values[i, j]
        
        reward, new_state = transition_and_reward(i, j, policy[i, j])

        values[i, j] = reward + discount * values[new_state]
        delta = max(delta, abs(tmp - values[i, j]))

    if delta < threshold:
      break

def policy_improvement():

  while True:
    policy_stable = True

    for i in range(4):
      for j in range(4):
        tmp = policy[i, j]
        
        next_states = possible_states(i, j)

        val = 0
        idx = 0
        for i, state in enumerate(next_states):
          reward, next_state = transition_and_reward(i, j, tmp) 
          tmp_val = reward + discount * values[next_state]

          if tmp_val > val:
            val = tmp_val
            idx = i

        policy[i, j] = idx

        if tmp != policy[i, j]:
          policy_stable = False

    if policy_stable:
      break



def transition_and_reward(x, y, direction):
  if direction == LEFT:
    x -= 1
  elif direction == RIGHT:
    x += 1
  elif direction == UP:
    y += 1
  elif direction == DOWN:
    y -= 1

  if (x, y) in holes:
    return 0, start
  elif (x, y) == finish:
    return 1, start
  else:
    return 0, (x, y)

def possible_states(x, y):

  states = [(x-1, y), (x+1, y), (x, y+1), (x, y-1)]

  if x == 0:
    states.remove((x-1, y))
  if x == 3:
    states.remove((x+1, y))
  if y == 0:
    states.remove((x, y-1))
  if y == 3:
    states.remove((x, y+1))

  return states

policy_eval()
policy_improvement()

print(values)
print(policy)