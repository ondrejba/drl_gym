import subprocess

learning_rates = [1e-3, 1e-4, 1e-5]
update_rates = [1e-2, 1e-3, 1e-4]
builds = ["multiple", "hydra"]

for learning_rate in learning_rates:
  for update_rate in update_rates:
    for build in builds:
      subprocess.call(["python", "bipedalwalker-naf.py", "--learning-rate", str(learning_rate),
                       "--update-rate", str(update_rate), "--build", build, "--monitor-frequency", "0"])