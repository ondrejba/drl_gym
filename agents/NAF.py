import tensorflow as tf
from tensorflow.core.framework import summary_pb2

import utils.architect as architect
from utils.ReplayBuffer import ReplayBuffer
import utils.utils as utils

class NAF:

  MODEL_NAME = "NAF"
  TARGET_MODEL_NAME = "target-NAF"

  def __init__(self, prep, state_dim, action_dim, buffer_size=10000, batch_size=32, steps_before_train=10000,
               train_freq=1, max_iters=1000000, max_reward=None, detailed_summary=False):

    self.prep = prep
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.detailed_summary = detailed_summary

    self.discount = 0.99
    self.learning_rate = 1e-4
    self.target_update_rate = 1e-3
    self.buffer_size = buffer_size
    self.batch_size = batch_size
    self.steps_before_train = steps_before_train
    self.train_freq = train_freq
    self.max_reward = max_reward
    self.max_iters = max_iters

    self.step = 0
    self.solved = False

    self.state_layers = [
      64, 32
    ]

    self.mu_layers = [
      16,
      self.action_dim
    ]

    self.l_layers = [
      16,
      (self.action_dim * (self.action_dim + 1)) / 2
    ]

    self.v_layers = [
      16,
      1
    ]

    self.action_inputs = None
    self.reward_inputs = None
    self.done = None
    self.state_inputs = None
    self.state_outputs = None
    self.mu_outputs = None
    self.l_outputs = None
    self.value_outputs = None
    self.next_state_inputs = None
    self.next_state_outputs = None
    self.target_value_outputs = None
    self.target = None
    self.advantages = None
    self.q_values = None
    self.loss = None
    self.global_step = None
    self.inc_global_step = None
    self.train_op = None
    self.target_update = None

    self.buffer = ReplayBuffer(buffer_size, self.state_dim, self.action_dim)

    self.build()

    self.merged = tf.summary.merge_all()

    self.session = tf.Session()

    self.summary_dir = utils.new_summary_dir(self.summary_dir)
    self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

    self.saver = tf.train.Saver(max_to_keep=None)

    init_op = tf.global_variables_initializer()
    self.session.run(init_op)

  def build(self):
    self.action_inputs = tf.placeholder(tf.float32, (None, self.action_dim))
    self.reward_inputs = tf.placeholder(tf.float32, (None, 1))
    self.done = tf.placeholder(tf.float32, (None, 1))

    self.state_inputs, self.state_outputs, self.mu_outputs, self.l_outputs, self.value_outputs = \
      self.build_network(self.MODEL_NAME)

    self.next_state_inputs, self.next_state_outputs, _, _, self.target_value_outputs = \
      self.build_network(self.TARGET_MODEL_NAME)

    self.target = self.reward_inputs + (1 - self.done) * self.discount * self.target_value_outputs

    # taken from https://github.com/carpedm20/NAF-tensorflow/blob/master/src/network.py
    pivot = 0
    rows = []
    for idx in range(self.action_dim):
      count = self.action_dim - idx

      diag_elem = tf.exp(tf.slice(self.l_outputs, (0, pivot), (-1, 1)))
      non_diag_elems = tf.slice(self.l_outputs, (0, pivot + 1), (-1, count - 1))
      row = tf.pad(tf.concat((diag_elem, non_diag_elems), 1), ((0, 0), (idx, 0)))
      rows.append(row)

      pivot += count

    L = tf.transpose(tf.stack(rows, axis=1), (0, 2, 1))
    P = tf.matmul(L, tf.transpose(L, (0, 2, 1)))

    adv_term = tf.expand_dims(self.action_inputs - self.mu_outputs, -1)
    self.advantages = -tf.matmul(tf.transpose(adv_term, [0, 2, 1]), tf.matmul(P, adv_term)) / 2
    self.advantages = tf.reshape(self.advantages, [-1, 1])

    self.q_values = self.advantages + self.value_outputs

    self.loss = tf.reduce_mean(architect.huber_loss(self.q_values - tf.stop_gradient(self.target)))

    tf.summary.scalar("training_loss", self.loss)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.inc_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1))

    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.train_op = optimizer.minimize(self.loss)

    self.create_target_update_op()

  def build_network(self, name):

    detailed_summary = self.detailed_summary
    if name == self.TARGET_MODEL_NAME:
      detailed_summary = False

    with tf.variable_scope(name):

      state_inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim))
      state_outputs = architect.dense_block(state_inputs, self.state_layers, name="state_branch", detailed_summary=detailed_summary)

      mu_outputs = architect.dense_block(state_outputs, self.mu_layers, "mu_branch", detailed_summary=detailed_summary)
      l_outputs = architect.dense_block(state_outputs, self.l_layers, "l_branch", detailed_summary=detailed_summary)
      value_outputs = architect.dense_block(state_outputs, self.v_layers, "value_branch", detailed_summary=detailed_summary)

      return state_inputs, state_outputs, mu_outputs, l_outputs, value_outputs

  def create_target_update_op(self):
    # inspired by: https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/neural_q_learner.py
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.MODEL_NAME)
    target_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.TARGET_MODEL_NAME)

    self.target_update = []
    for v_source, v_target in zip(net_vars, target_net_vars):
      # this is equivalent to target = (1-alpha) * target + alpha * source
      update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
      self.target_update.append(update_op)

    self.target_update = tf.group(*self.target_update)

  def learn(self):
    # learn
    batch = self.buffer.sample(self.batch_size)

    merged, _ = self.session.run([self.merged, self.train_op], feed_dict={
      self.state_inputs: batch["states"],
      self.action_inputs: batch["actions"],
      self.reward_inputs: batch["rewards"],
      self.next_state_inputs: batch["next_states"],
      self.done: batch["done"]
    })

    self.summary_writer.add_summary(merged, global_step=self.step)

    # target update
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
        action = self.mu_outputs

      tmp_state = state
      tmp_skip = skip

      state, reward, done, info = env.step(action)
      state, skip = self.prep.process(state)

      total_reward += reward

      if not tmp_skip and not tmp_skip:
        self.buffer.add({
            "state": tmp_state[0],
            "action": action,
            "reward": reward,
            "next_state": state[0],
            "done": int(done)
          })

      if self.step >= self.steps_before_train and self.step % self.train_freq == 0 and not self.solved:
        # learn
        self.learn()

      _, self.step = self.session.run([self.inc_global_step, self.global_step])

      if done:
        break

    summary_value = summary_pb2.Summary.Value(tag="episode_reward", simple_value=total_reward)
    summary_2 = summary_pb2.Summary(value=[summary_value])
    self.summary_writer.add_summary(summary_2, global_step=self.step)

    if self.max_reward is not None:
      if total_reward >= self.max_reward:
        self.solved = True
      else:
        self.solved = False

    if self.step == self.max_iters:
      self.saver.save(self.session, self.summary_dir, global_step=self.step)

    return total_reward, self.step

  def close(self):
    self.session.close()