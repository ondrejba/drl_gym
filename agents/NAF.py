import tensorflow as tf

import agents.architect as architect

class NAF():

  MODEL_NAME = "NAF"
  TARGET_MODEL_NAME = "target-NAF"

  def __init__(self, state_dim, action_dim, detailed_summary=False):

    self.state_dim = state_dim
    self.action_dim = action_dim
    self.detailed_summary = detailed_summary

    self.discount = 0.9
    self.learning_rate = 1e-04
    self.target_update_rate = 1e-2

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

  def build(self):
    self.action_inputs = tf.placeholder(tf.float32, (None, self.action_dim))
    self.reward_inputs = tf.placeholder(tf.float32, (None, 1))

    self.state_inputs, self.state_outputs, self.mu_outputs, self.l_outputs, self.value_outputs = \
      self.build_network(self.MODEL_NAME)

    self.next_state_inputs, self.next_state_outputs, _, _, self.target_value_outputs = \
      self.build_network(self.TARGET_MODEL_NAME)

    self.target = self.reward_inputs + self.discount * self.target_value_outputs

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