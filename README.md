## Deep Reinforcement Learning gym

This repository contains a modular framework for testing Deep Reinforcement Learning algorithms in 
[OpenAI's Gym](http://gym.openai.com/). All the agents are implemented in [Tensorflow](https://www.tensorflow.org/).

### Requirements ###

* Python >= 3 - if you want to use [Mujoco](https://gym.openai.com/envs#mujoco), you will need to install Python 3.5.2 (as of 23.7.2017)
* OpenAI gym
* Tensorflow and other packages listed in **requirements.txt**

#### Mujoco ####

Mujoco is required only if you want to test continuous control tasks like Reacher-v1. Installing Mujoco for Python
is quite difficult as of 23.7.2017. Your best bet is to follow this [thread](https://github.com/openai/mujoco-py/issues/47).
Be advised that installing *xserver-xorg-dev* might stop Ubuntu from receiving keyboard and mouse input.

### Usage ###

You can run continuous control task using **train_continuous_task.py** and discrete control tasks with **train_discrete_task.py**.

Both of the scripts take the following four arguments:

1. environment name
2. agent name
3. state preprocessing
4. exploration policy

#### State preprocessing ####

Most of the environments don't need any specific state preprocessing. Pass the value **simple** in this case.

On the other hand, environments like **Atari** are easier to solve if several frames are stacked on top of each other
and preprocessed. In that case, pass the value **atari**.

### Contribute ###

You can contribute by create a new agent.

Each agent should have the following public methods:

* **learn**: perform a single training step
* **act**: generate an action given a state
* **perceive**: save a transition into the replay buffer (only Temporal Difference methods are supported for now)
* **log_scalar**: log scalar to a tensorboard summary; you can leave this method empty
* **save**: save the agent
* **close**: close the agent's session
