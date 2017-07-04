import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
import math, os, time

import agents.utils as utils
import agents.policy as policy
import agents.architect as architect
from agents.ReplayBuffer import ReplayBuffer


class DeepQNetwork():

  def __init__(self, state_dim, action_dim, name, layers=None, learning_rate=1e-3, update_frequency=500,
               buffer_size=50000, batch_size=2, exploration=0.1, lin_exp_end_iter=None, lin_exp_final_eps=None, max_iters=200000, discount=0.9,
               use_huber_loss=True, detailed_summary=False, max_reward=200):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.discount = discount
    self.summary_dir = os.path.join(name, "summary")
    self.use_huber_loss = use_huber_loss
    self.detailed_summary = detailed_summary

    if layers:
      self.layers = layers
    else:
      self.layers = [
        64,
        32,
        self.action_dim
      ]

    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.update_frequency = update_frequency
    self.epsilon = exploration
    self.lin_exp_end_iter = lin_exp_end_iter
    self.lin_exp_final_eps = lin_exp_final_eps
    self.max_iters = max_iters
    self.step = 0
    self.steps_before_learn = 1000
    self.solved = False
    self.max_reward = max_reward

    self.build_all()
  
    self.merged = tf.summary.merge_all()

    self.session = tf.Session()
    self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

    init_op = tf.global_variables_initializer()
    self.session.run(init_op)

    self.buffer = ReplayBuffer(buffer_size, self.state_dim, self.action_dim)

  def build_all(self):

    self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
    self.done = tf.placeholder(tf.float32, (None,), name="done")

    self.build_network()
    self.build_target_network()
    self.create_hard_target_update_op()

    self.action_q_values = tf.reduce_sum(self.q_values * self.actions, axis=1)
    self.max_target_q_values = tf.reduce_max(self.target_q_values, axis=1)

    self.targets = self.rewards + (1 - self.done) * (self.discount * self.max_target_q_values)

    if self.detailed_summary:
      architect.variable_summaries(self.targets, name="targets")

    td_diff = self.action_q_values - tf.stop_gradient(self.targets)

    if self.use_huber_loss:
      loss = tf.reduce_mean(architect.huber_loss(td_diff))
    else:
      loss = tf.reduce_mean(tf.pow(td_diff, 2))

    tf.summary.scalar("loss", loss)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.inc_global_step = tf.assign(self.global_step, self.global_step + 1)
    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=self.global_step)

  def build_network(self):

    with tf.variable_scope("network"):

      self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
      self.actions = tf.placeholder(tf.float32, (None, self.action_dim), name="actions")
      self.q_values = architect.dense_block(self.states, self.layers, "q-network", detailed_summary=self.detailed_summary)

  def build_target_network(self):

    with tf.variable_scope("target_network"):

      self.next_states = tf.placeholder(tf.float32, (None, self.state_dim), name="next_states")
      self.target_q_values = architect.dense_block(self.next_states, self.layers, "q-network")

      if self.detailed_summary:
        architect.variable_summaries(self.target_q_values, name="target_q_values")

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

  def run_episode(self, env):

    state = env.reset()
    total_reward = 0

    while True:
      # play
      state_vec = np.expand_dims(state, axis=0)
      q_values = self.session.run(self.q_values, feed_dict={self.states: state_vec})[0]

      if self.lin_exp_end_iter:
        if self.lin_exp_final_eps is None:
          fin_eps = 0
        else:
          fin_eps = self.lin_exp_final_eps

        self.epsilon = policy.linear_schedule(self.lin_exp_end_iter, self.max_iters, self.step, final_eps=fin_eps)

      action = policy.epsilon_greedy(q_values, self.epsilon)

      action_one_hot = np.zeros(self.action_dim)
      action_one_hot[action] = 1

      tmp_state = state
      state, reward, done, info = env.step(action)

      total_reward += reward

      self.buffer.add({
          "state": tmp_state,
          "action": action_one_hot,
          "reward": reward,
          "next_state": state,
          "done": int(done)
        })

      if self.step > self.steps_before_learn and not self.solved:
        # learn
        batch = self.buffer.sample(self.batch_size)

        merged, self.step, _ = self.session.run([self.merged, self.global_step, self.train_op], feed_dict={
            self.states: batch["states"],
            self.actions: batch["actions"],
            self.rewards: batch["rewards"],
            self.next_states: batch["next_states"],
            self.done: batch["done"]
          })

        self.summary_writer.add_summary(merged, global_step=self.step)

        # target update
        if self.step % self.update_frequency == 0:
          self.session.run(self.target_update)
      else:
        _, self.step = self.session.run([self.inc_global_step, self.global_step])

      if done:
        break

    summary_value = summary_pb2.Summary.Value(tag="episode_reward", simple_value=total_reward)
    summary_2 = summary_pb2.Summary(value=[summary_value])
    self.summary_writer.add_summary(summary_2, global_step=self.step)

    if total_reward >= self.max_reward:
      self.solved = True
    else:
      self.solved = False

    return total_reward, self.step

  def close(self):
    self.session.close()