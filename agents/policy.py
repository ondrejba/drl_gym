import random
import numpy as np

def epsilon_greedy(probabilities, epsilon):
  rand = random.random()

  if rand <= epsilon:
    return random.randint(0, len(probabilities) - 1)
  else:
    return np.argmax(probabilities)