import os
import numpy as np

def sample(probs):
  return np.random.choice(len(probs), None, p=probs)

def new_summary_dir(summary_dir):
  i = 1
  while os.path.isdir(os.path.join(summary_dir, "run{}".format(i))):
    i += 1

  summary_dir = os.path.join(summary_dir, "run{}".format(i))
  os.mkdir(summary_dir)
  return summary_dir

def log_params(summary_dir, params):

  with open(os.path.join(summary_dir, "params.txt"), "w") as file:
    for key in params.keys():
      file.write("{}: {}\n".format(key, params[key]))

class MonitorCallable:

  def __init__(self, video_ep_freq=100):
    self.video_ep_freq = video_ep_freq

  def call(self, idx):
    if self.video_ep_freq == 0:
      return False
    else:
      return (idx != 0) and (idx % self.video_ep_freq == 0)

class Timer:

  def __init__(self):
    self.times = {}
    self.counts = {}

  def add(self, name, value):

    if name in self.times:
      self.times[name] += value
      self.counts[name] += 1
    else:
      self.times[name] = value
      self.counts[name] = 1

  def reset(self, name):
    self.times[name] = 0
    self.counts[name] = 0

  def reset_all(self):
    for key in self.times.keys():
      self.times[key] = 0
      self.counts[key] = 0

  def print(self):
    for key in self.times.keys():
      print("%s: %.8f" % (key, (self.times[key] / self.counts[key])))