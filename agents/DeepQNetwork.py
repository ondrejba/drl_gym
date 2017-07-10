import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
import os

import utils.policy as policy
import utils.architect as architect
from utils.ReplayBuffer import ReplayBuffer
import utils.utils as utils

class DeepQNetwork:

  ACTION_VALUE_NET_NAME = "q-network"
  TARGET_ACTION_VALUE_NET_NAME = "target-q-network"

  def __init__(self, network, prep, exp_policy, state_dim, action_dim, name, learning_rate=1e-3,
               hard_update_frequency=500, soft_update_rate=None, buffer_size=50000, batch_size=32, num_steps=200000,
               discount=0.99, use_huber_loss=True, detailed_summary=False, max_reward=200, steps_before_learn=1000,
               train_freq=1, save_end=True):

    self.network = network
    self.prep = prep
    self.exp_policy = exp_policy
    self.greedy_policy = policy.Greedy()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.discount = discount
    self.summary_dir = os.path.join(name, "summary")
    self.use_huber_loss = use_huber_loss
    self.detailed_summary = detailed_summary

    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.hard_update_frequency = hard_update_frequency
    self.soft_update_rate = soft_update_rate
    self.num_steps = num_steps
    self.step = 0
    self.steps_before_learn = steps_before_learn
    self.train_freq = train_freq
    self.solved = False
    self.max_reward = max_reward
    self.save_end = save_end

    self.actions = None
    self.rewards = None
    self.done = None
    self.action_q_values = None
    self.max_target_q_values = None
    self.targets = None
    self.global_step = None
    self.inc_global_step = None
    self.train_op = None
    self.states = None
    self.q_values = None
    self.next_states = None
    self.target_q_values = None
    self.target_update = None

    self.build_all()
  
    self.merged = tf.summary.merge_all()

    self.session = tf.Session()

    self.summary_dir = utils.new_summary_dir(self.summary_dir)
    self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

    self.saver = tf.train.Saver(max_to_keep=None)

    init_op = tf.global_variables_initializer()
    self.session.run(init_op)

    self.buffer = ReplayBuffer(buffer_size, self.state_dim, self.action_dim)

  def build_all(self):

    self.actions = tf.placeholder(tf.float32, (None, self.action_dim), name="actions")
    self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
    self.done = tf.placeholder(tf.float32, (None,), name="done")

    self.build_network()
    self.build_target_network()

    if self.soft_update_rate is not None:
      self.create_soft_target_update_op()
    else:
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
    self.inc_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1))
    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

  def build_network(self):
    self.states, self.q_values = self.network.build(self.state_dim, self.action_dim, self.ACTION_VALUE_NET_NAME)

  def build_target_network(self):
    self.next_states, self.target_q_values = self.network.build(self.state_dim, self.action_dim, self.TARGET_ACTION_VALUE_NET_NAME)

  def create_soft_target_update_op(self):
    # inspired by: https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/neural_q_learner.py
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ACTION_VALUE_NET_NAME)
    target_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.TARGET_ACTION_VALUE_NET_NAME)

    self.target_update = []
    for v_source, v_target in zip(net_vars, target_net_vars):
      # this is equivalent to target = (1-alpha) * target + alpha * source
      update_op = v_target.assign_sub(self.soft_update_rate * (v_target - v_source))
      self.target_update.append(update_op)

    self.target_update = tf.group(*self.target_update)

  def create_hard_target_update_op(self):
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ACTION_VALUE_NET_NAME)
    target_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.TARGET_ACTION_VALUE_NET_NAME)

    self.target_update = []
    for v_source, v_target in zip(net_vars, target_net_vars):
      update_op = v_target.assign(v_source)
      self.target_update.append(update_op)

    self.target_update = tf.group(*self.target_update)

  def learn(self):
    # learn
    batch = self.buffer.sample(self.batch_size)

    merged, _ = self.session.run([self.merged, self.train_op], feed_dict={
      self.states: batch["states"],
      self.actions: batch["actions"],
      self.rewards: batch["rewards"],
      self.next_states: batch["next_states"],
      self.done: batch["done"]
    })

    self.summary_writer.add_summary(merged, global_step=self.step)

    # target update
    if self.soft_update_rate is not None:
      self.session.run(self.target_update)
    elif self.step % self.hard_update_frequency == 0:
      self.session.run(self.target_update)

  def run_episode(self, env):

    state = env.reset()
    state, skip = self.prep.process(state)

    total_reward = 0

    while True:
      # play
      if skip:
        action = env.action_space.sample()
      else:
        q_values = self.session.run(self.q_values, feed_dict={self.states: state})[0]

        if self.solved:
          action = self.greedy_policy.select_action(q_values)
        else:
          action = self.exp_policy.select_action(q_values)

      action_one_hot = np.zeros(self.action_dim)
      action_one_hot[action] = 1

      tmp_state = state
      tmp_skip = skip

      state, reward, done, info = env.step(action)
      state, skip = self.prep.process(state)

      total_reward += reward

      if not tmp_skip and not tmp_skip:
        self.buffer.add({
            "state": tmp_state[0],
            "action": action_one_hot,
            "reward": reward,
            "next_state": state[0],
            "done": int(done)
          })

      if self.step >= self.steps_before_learn and self.step % self.train_freq == 0 and not self.solved:
        # learn
        self.learn()

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

    if self.step == self.num_steps:
      self.saver.save(self.session, self.summary_dir, global_step=self.step)

    return total_reward, self.step

  def close(self):
    self.session.close()