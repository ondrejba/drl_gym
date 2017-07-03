import random
import numpy as np

def epsilon_greedy(probabilities, epsilon):
  rand = random.random()

  if rand <= epsilon:
    return random.randint(0, len(probabilities) - 1)
  else:
    return np.argmax(probabilities)

def linear_schedule(fract_iters_end, max_iters, current_iter, final_eps=0):
  return max(final_eps, 1 - current_iter / (fract_iters_end * max_iters))