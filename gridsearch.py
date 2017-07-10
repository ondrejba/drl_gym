import subprocess

"""
learning_rates = [1e-3, 1e-4, 1e-5]
update_rates = [1e-2, 1e-3, 1e-4]
builds = ["hydra"]

num_steps = 50000

for learning_rate in learning_rates:
  for update_rate in update_rates:
    for build in builds:
      subprocess.call(["python", "train_continuous_task.py", "Pendulum-v0", "naf", "simple", "none", "--build", build,
                       "--learning-rate", str(learning_rate), "--update-rate", str(update_rate), "--num-steps",
                       str(num_steps), "--monitor-frequency", str(0)])
"""

for i in range(3):
  subprocess.call(["python", "train_continuous_task.py", "Pendulum-v0", "naf", "simple", "none", "--build", "hydra",
                         "--learning-rate", str(1e-3), "--update-rate", str(1e-2), "--num-steps",
                         str(50000), "--monitor-frequency", str(0)])