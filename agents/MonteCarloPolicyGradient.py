import tensorflow as tf
import numpy as np
import math, os

import utils.utils as utils

class MonteCarloPolicyGradient:

  def __init__(self, state_dim, action_dim, name, discount=0.9):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.discount = discount
    self.summary_dir = os.path.join(name, "summary")

    self.policy_layers = [
      32,
      16,
      self.action_dim
    ]

    self.value_layers = [
      32,
      16,
      1
    ]

    self.policy_learning_rate = 1e-4
    self.value_learning_rate = 1e-4

    self.policy_gradient()
    self.value_gradient()

    self.episode_return = tf.placeholder(tf.float32)
    self.episode_return_summary = tf.summary.scalar("episode_return", self.episode_return)

    self.session = tf.Session()
    self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

    init_op = tf.global_variables_initializer()
    self.session.run(init_op)

  def policy_gradient(self):
    with tf.variable_scope("policy"):

      self.policy_state = tf.placeholder(tf.float32, (None, self.state_dim), name="state")
      self.policy_actions = tf.placeholder(tf.float32, (None, self.action_dim), name="actions")
      self.policy_advantages = tf.placeholder(tf.float32, (None, 1), name="advantages")

      output = self.policy_state
      for i, layer in enumerate(self.policy_layers):
        if i == len(self.policy_layers) - 1:
          output = tf.layers.dense(output, layer)
        else:
          output = tf.layers.dense(output, layer, activation=tf.nn.relu)

      self.policy_probs = tf.nn.softmax(output)
      good_probs = tf.reduce_sum(self.policy_probs * self.policy_actions, reduction_indices=[1])

      eligibility = tf.log(good_probs) * self.policy_advantages
      loss = - tf.reduce_sum(eligibility)

      loss_summary = tf.summary.scalar("policy_loss", loss)
      self.policy_summary = tf.summary.merge([loss_summary])

      self.policy_optimizer = tf.train.AdamOptimizer(self.policy_learning_rate).minimize(loss)

  def value_gradient(self):
    with tf.variable_scope("value"):

      self.value_state = tf.placeholder(tf.float32, (None, self.state_dim), name="state")
      self.target_values = tf.placeholder(tf.float32, (None, 1), name="target_values")
      
      output = self.value_state
      for i, layer in enumerate(self.value_layers):
        if i == len(self.policy_layers) - 1:
          output = tf.layers.dense(output, layer)
        else:
          output = tf.layers.dense(output, layer, activation=tf.nn.relu)

      self.values = output

      diffs = self.values - self.target_values
      loss = tf.nn.l2_loss(diffs)

      loss_summary = tf.summary.scalar("value_loss", loss)
      self.value_summary = tf.summary.merge([loss_summary])

      self.value_optimizer = tf.train.AdamOptimizer(self.value_learning_rate).minimize(loss)

  def run_episode(self, env, step):

    state = env.reset()
    total_reward = 0

    states = []
    actions = []
    advantages = []
    transitions = []

    update_vals = []
    future_rewards = []

    i = 0
    while True:

      state_vec = np.expand_dims(state, axis=0)
      probs = self.session.run(self.policy_probs, feed_dict = {self.policy_state: state_vec})
      action = utils.sample(probs[0])

      states.append(state)
      action_blank = np.zeros(self.action_dim)
      action_blank[action] = 1
      actions.append(action_blank)

      tmp_state = state
      state, reward, done, info = env.step(action)
      transitions.append((tmp_state, action, reward))
      total_reward += reward

      for j in range(i + 1):
        discount = math.pow(0.9, i - j)
        value = discount * reward

        if len(future_rewards) <= j:
          future_rewards.append(value)
        else:
          future_rewards[j] += value

      if done:
        break

      i += 1

    future_rewards = future_rewards[:i + 1]

    values = self.session.run(self.values, feed_dict={self.value_state: states})

    for index, trans in enumerate(transitions):
      obs, action, reward = trans
      advantages.append(future_rewards[index] - values[index])

    target_vals_vector = np.expand_dims(future_rewards, axis=1)

    value_summary, _ = self.session.run([self.value_summary, self.value_optimizer], feed_dict={self.value_state: states, self.target_values: target_vals_vector})
    policy_summary, _ = self.session.run([self.policy_summary, self.policy_optimizer], feed_dict={self.policy_state: states, self.policy_advantages: advantages, self.policy_actions: actions})
    return_summary = self.session.run(self.episode_return_summary, feed_dict={self.episode_return: total_reward})

    self.summary_writer.add_summary(value_summary, step)
    self.summary_writer.add_summary(policy_summary, step)
    self.summary_writer.add_summary(return_summary, step)

    return total_reward

  def close(self):
    self.session.close()