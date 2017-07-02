import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
import math, os, time

import agents.utils as utils
import agents.policy as policy
from agents.ReplayBuffer import ReplayBuffer


class DeepQNetwork():

  def __init__(self, state_dim, action_dim, name, discount=0.9):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.discount = discount
    self.summary_dir = os.path.join(name, "summary")

    self.layers = [
      64,
      32,
      self.action_dim
    ]

    self.learning_rate = 1e-4
    self.batch_size = 64
    self.update_frequency = 1000
    self.epsilon = 0.05

    self.build_all()
  
    self.merged = tf.summary.merge_all()

    self.session = tf.Session()
    self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

    init_op = tf.global_variables_initializer()
    self.session.run(init_op)

    self.buffer = ReplayBuffer(10000, self.state_dim, self.action_dim)

  def build_all(self):

    self.rewards = tf.placeholder(tf.float32, (None, 1), name="rewards")

    self.build_network()
    self.build_target_network()
    self.create_hard_target_update_op()

    self.action_q_values = self.q_values * self.actions
    self.action_target_q_values = self.target_q_values * self.actions
    self.action_rewards = self.rewards * self.actions

    self.targets = self.action_rewards + self.discount * self.action_target_q_values
    loss = tf.reduce_mean(tf.pow(self.action_q_values - self.targets, 2))

    tf.summary.scalar("loss", loss)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=self.global_step)

  def build_network(self):

    with tf.variable_scope("network"):

      self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
      self.actions = tf.placeholder(tf.float32, (None, self.action_dim), name="actions")

      output = self.states
      for i, layer in enumerate(self.layers):
        if i == len(self.layers) - 1:
          output = tf.layers.dense(output, layer)
        else:
          output = tf.layers.dense(output, layer, activation=tf.nn.relu)

      self.q_values = output
      self.action = tf.argmax(self.q_values, axis=1)

  def build_target_network(self):

    with tf.variable_scope("target_network"):

      self.next_states = tf.placeholder(tf.float32, (None, self.state_dim), name="next_states")

      output = self.next_states
      for i, layer in enumerate(self.layers):
        if i == len(self.layers) - 1:
          output = tf.layers.dense(output, layer)
        else:
          output = tf.layers.dense(output, layer, activation=tf.nn.relu)

      self.target_q_values = output

  def create_soft_target_update_op(self):
    # inspired by: https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/neural_q_learner.py
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")
    target_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")

    self.target_update = []
    for v_source, v_target in zip(net_vars, target_net_vars):
      # this is equivalent to target = (1-alpha) * target + alpha * source
      update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
      self.target_update.append(update_op)

    self.target_update = tf.group(*self.target_update)

  def create_hard_target_update_op(self):
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")
    target_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")

    self.target_update = []
    for v_source, v_target in zip(net_vars, target_net_vars):
      update_op = v_target.assign(v_source)
      self.target_update.append(update_op)

    self.target_update = tf.group(*self.target_update)

  def run_episode(self, env, step):

    state = env.reset()
    total_reward = 0

    i = 0
    timer = utils.Timer()
    ep_start = time.time()
    while True:

      # play

      play_start = time.time()
      state_vec = np.expand_dims(state, axis=0)
      q_values = self.session.run(self.q_values, feed_dict={self.states: state_vec})[0]

      action = policy.epsilon_greedy(q_values, self.epsilon)

      action_one_hot = np.zeros(self.action_dim)
      action_one_hot[action] = 1

      tmp_state = state
      state, reward, done, info = env.step(action)
      total_reward += reward

      if done:
        state = np.zeros(tmp_state.shape)

      self.buffer.add({
          "state": tmp_state,
          "action": action_one_hot,
          "reward": reward,
          "next_state": state
        })
      timer.add("play", time.time() - play_start)

      if self.buffer.get_size() >= self.batch_size:
        # learn

        sample_start = time.time()
        batch = self.buffer.sample(self.batch_size)
        timer.add("sample", time.time() - sample_start)

        learning_start = time.time()
        merged, global_step, _ = self.session.run([self.merged, self.global_step, self.train_op], feed_dict={
            self.states: batch["states"],
            self.actions: batch["actions"],
            self.rewards: batch["rewards"],
            self.next_states: batch["next_states"]
          })
        timer.add("learning", time.time() - learning_start)

        update_start = time.time()
        self.summary_writer.add_summary(merged, global_step=global_step)

        # target update
        if (global_step * self.batch_size) % self.update_frequency == 0:
          self.session.run(self.target_update)
        timer.add("update", time.time() - update_start)

      if done:
        break

      i += 1

    timer.add("episode", time.time() - ep_start)

    sum_start = time.time()

    summary_value = summary_pb2.Summary.Value(tag="episode_reward", simple_value=total_reward)
    summary_2 = summary_pb2.Summary(value=[summary_value])
    self.summary_writer.add_summary(summary_2, global_step=step)

    timer.add("last_summary", time.time() - sum_start)
    #timer.print()

    return total_reward

  def close(self):
    self.session.close()