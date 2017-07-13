import tensorflow as tf
import math, os
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.core.framework import summary_pb2
import numpy as np

import utils.utils as utils
import utils.architect as architect
from utils.ReplayBuffer import ReplayBuffer

class DDPG:

  CRITIC_NAME = "critic"
  TARGET_CRITIC_NAME = "target_critic"

  ACTOR_NAME = "actor"
  TARGET_ACTOR_NAME = "target_actor"

  def __init__(self, prep, policy, state_dim, action_dim, action_high, action_low, monitor_directory, actor_learning_rate=1e-4, critic_learning_rate=1e-3,
               critic_target_update_rate=1e-3, actor_target_update_rate=1e-3, discount=0.99,
               l2_decay=1e-2, buffer_size=1000000, steps_before_train=10000, max_reward=None, train_freq=1,
               num_steps=500000, batch_size=64, detail_summary=False, tanh_action=True, batch_norm=False):

    self.prep = prep
    self.policy = policy
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.action_high = action_high
    self.action_low = action_low

    self.critic_learning_rate = critic_learning_rate
    self.actor_learning_rate = actor_learning_rate
    self.critic_target_update_rate = critic_target_update_rate
    self.actor_target_update_rate = actor_target_update_rate
    self.discount = discount
    self.batch_size = batch_size
    self.l2_decay = l2_decay
    self.buffer_size = buffer_size
    self.steps_before_train = steps_before_train
    self.max_reward = max_reward
    self.train_freq = train_freq
    self.num_steps = num_steps
    self.summary_dir = os.path.join(monitor_directory, "summary")
    self.detail_summary = detail_summary
    self.tanh_action = tanh_action
    self.batch_norm = batch_norm

    self.step = 0
    self.solved = False

    self.buffer = ReplayBuffer(buffer_size, self.state_dim, self.action_dim)

    self.build()

    self.summary_dir = utils.new_summary_dir(self.summary_dir)
    utils.log_params(self.summary_dir, {
      "actor learning rate": self.actor_learning_rate,
      "critic learning rate": self.critic_learning_rate,
      "batch size": self.batch_size,
      "actor update rate": self.actor_target_update_rate,
      "critic update rate": self.critic_target_update_rate,
      "buffer size": self.buffer_size,
      "train frequency": self.train_freq
    })

    self.saver = tf.train.Saver(max_to_keep=None)

    init_op = tf.global_variables_initializer()
    self.session = tf.Session()

    self.merged = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

    self.session.run(init_op)

  def build_critic(self, name, state_input, action_input):

    bn_training = self.is_training
    if name == self.TARGET_CRITIC_NAME:
      bn_training = False

    with tf.variable_scope(name):

      # weights and biases
      W1 = self.get_weights((self.state_dim, 400), self.state_dim, name="W1")
      b1 = self.get_weights((400,), self.state_dim, name="b1")

      W2 = self.get_weights((400, 300), 400 + self.action_dim, name="W2")
      b2 = self.get_weights((300,), 400 + self.action_dim, name="b2")

      W2_action = self.get_weights((self.action_dim, 300), 400 + self.action_dim, name="W2_action")

      W3 = tf.Variable(tf.random_uniform((300, 1), -3e-3, 3e-3), name="W3")
      b3 = tf.Variable(tf.random_uniform((1,), -3e-3, 3e-3), name="b3")

      # layers
      if self.batch_norm:
        state_input = tf.layers.batch_normalization(state_input, training=bn_training, momentum=0.9, epsilon=1e-5)

      layer_1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
      if self.batch_norm:
        layer_1 = tf.layers.batch_normalization(layer_1, training=bn_training, momentum=0.9, epsilon=1e-5)

      layer_2 = tf.nn.relu(tf.matmul(layer_1, W2) + tf.matmul(action_input, W2_action) + b2)

      output_layer = tf.matmul(layer_2, W3) + b3

      # summary
      if self.detail_summary and name == self.CRITIC_NAME:
        self.critic_summaries = [
          architect.variable_summaries(W1, "W1"),
          architect.variable_summaries(b1, "b1"),

          architect.variable_summaries(W2, "W2"),
          architect.variable_summaries(b2, "b2"),
          architect.variable_summaries(W2_action, "W2_action"),

          architect.variable_summaries(W3, "W3"),
          architect.variable_summaries(b3, "b3"),

          architect.variable_summaries(layer_1, "layer_1"),
          architect.variable_summaries(layer_2, "layer_2"),
          architect.variable_summaries(output_layer, "output_layer")
        ]

      # weight decay
      weights = [W1, b1, W2, b2, W2_action, W3, b3]
      weight_decay = tf.add_n([self.l2_decay * tf.nn.l2_loss(var) for var in weights])

      return output_layer, weight_decay

  def build_actor(self, name, state_input):

    bn_training = self.is_training
    if name == self.TARGET_ACTOR_NAME:
      bn_training = False

    with tf.variable_scope(name):

      # weights and biases
      W1 = self.get_weights((self.state_dim, 400), self.state_dim, name="W1")
      b1 = self.get_weights((400,), self.state_dim, name="b1")

      W2 = self.get_weights((400, 300), 400, name="W2")
      b2 = self.get_weights((300,), 400, name="b2")

      W3 = tf.Variable(tf.random_uniform((300, self.action_dim), -3e-3, 3e-3), name="W3")
      b3 = tf.Variable(tf.random_uniform((self.action_dim,), -3e-3, 3e-3), name="b3")

      # layers
      if self.batch_norm:
        state_input = tf.layers.batch_normalization(state_input, training=bn_training, momentum=0.9, epsilon=1e-5)

      layer_1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
      if self.batch_norm:
        layer_1 = tf.layers.batch_normalization(layer_1, training=bn_training, momentum=0.9, epsilon=1e-5)

      layer_2 = tf.nn.relu(tf.matmul(layer_1, W2) + b2)
      if self.batch_norm:
        layer_2 = tf.layers.batch_normalization(layer_2, training=bn_training, momentum=0.9, epsilon=1e-5)

      output_layer = tf.matmul(layer_2, W3) + b3

      # summary
      if self.detail_summary and name == self.ACTOR_NAME:
        self.actor_summaries = [
          architect.variable_summaries(W1, "W1"),
          architect.variable_summaries(b1, "b1"),

          architect.variable_summaries(W2, "W2"),
          architect.variable_summaries(b2, "b2"),

          architect.variable_summaries(W3, "W3"),
          architect.variable_summaries(b3, "b3"),

          architect.variable_summaries(layer_1, "layer_1"),
          architect.variable_summaries(layer_2, "layer_2"),
          architect.variable_summaries(output_layer, "output_layer")
        ]

      if self.tanh_action:
        return tf.nn.tanh(output_layer)
      else:
        return output_layer

  def build(self):

    self.state_input = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="state_input")
    self.next_state_input = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="next_state_input")
    self.action_input = tf.placeholder(tf.float32, shape=(None, self.action_dim), name="action_input")
    self.reward_input = tf.placeholder(tf.float32, shape=(None,), name="reward_input")
    self.done_input = tf.placeholder(tf.float32, shape=(None,), name="done_input")
    self.is_training = tf.placeholder(tf.bool, name="is_training")

    # inputs summary
    if self.detail_summary:
      self.inputs_summaries = [
        architect.variable_summaries(self.state_input),
        architect.variable_summaries(self.next_state_input),
        architect.variable_summaries(self.action_input),
        architect.variable_summaries(self.reward_input),
        architect.variable_summaries(self.done_input)
      ]

    self.target_action = self.build_actor(self.TARGET_ACTOR_NAME, self.next_state_input)

    self.q_value, weight_decay = self.build_critic(self.CRITIC_NAME, self.state_input, self.action_input)
    self.target_q_value, _ = self.build_critic(self.TARGET_CRITIC_NAME, self.next_state_input, self.target_action)

    self.tmp = tf.expand_dims(self.reward_input, 1)

    self.targets = tf.expand_dims(self.reward_input, 1) + self.discount * (1 - tf.expand_dims(self.done_input, 1)) * self.target_q_value
    self.diff = self.targets - self.q_value

    if self.detail_summary:
      architect.variable_summaries(self.diff, name="diff")

    self.loss = tf.reduce_mean(tf.square(tf.stop_gradient(self.targets) - self.q_value)) + weight_decay
    self.loss_summary = tf.summary.scalar("critic_loss", self.loss)

    self.critic_train_op = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.loss)

    self.action = self.build_actor(self.ACTOR_NAME, self.state_input)
    self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ACTOR_NAME)
    self.action_gradients = tf.gradients(self.q_value, self.action_input)[0]
    self.actor_params_gradient = tf.gradients(self.action, self.actor_params, - self.action_gradients)

    # actor gradients summary
    if self.detail_summary:
      self.actor_summaries.append(architect.variable_summaries(self.action_gradients, name="action_gradient"))
      for grad in self.actor_params_gradient:
        self.actor_summaries.append(architect.variable_summaries(grad, name="actor_parameter_gradients"))

    self.actor_train_op = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(zip(self.actor_params_gradient, self.actor_params))

    self.target_critic_update = architect.create_target_update_ops(self.CRITIC_NAME, self.TARGET_CRITIC_NAME, self.critic_target_update_rate)
    self.target_actor_update = architect.create_target_update_ops(self.ACTOR_NAME, self.TARGET_ACTOR_NAME, self.actor_target_update_rate)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.inc_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1))

  @staticmethod
  def get_weights(shape, input_shape, name="var"):
    return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(input_shape), 1 / math.sqrt(input_shape)), name=name)

  def train_actor(self, states):

    actions = self.session.run(self.action, feed_dict={
      self.state_input: states,
      self.is_training: False
    })

    self.session.run(self.actor_train_op, feed_dict={
      self.state_input: states,
      self.action_input: actions,
      self.is_training: True
    })

  def train_critic(self, states, actions, rewards, next_states, done):
    _, targets, summary, tmp = self.session.run([self.critic_train_op, self.targets, self.merged, self.diff], feed_dict={
      self.state_input: states,
      self.action_input: actions,
      self.reward_input: rewards,
      self.next_state_input: next_states,
      self.done_input: done,
      self.is_training: True
    })
    self.summary_writer.add_summary(summary, global_step=self.step)

  def train_step(self, batch):
    self.train_critic(batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["done"])
    self.train_actor(batch["states"])
    self.session.run([self.target_critic_update, self.target_actor_update])

  def run_episode(self, env, eval=False, ep_idx=0, learn=True):

    self.policy.reset()

    state = env.reset()
    state, skip = self.prep.process(state)

    total_reward = 0

    while True:
      # play
      if skip:
        action = env.action_space.sample()
      else:
        action = self.session.run(self.action, feed_dict={
          self.state_input: state,
          self.is_training: False
        })[0]

        if not eval and not self.solved:
          action = self.policy.add_noise(action)

      action = np.clip(action, -1, 1)

      tmp_state = state
      tmp_skip = skip

      state, reward, done, _ = env.step(action)
      state, skip = self.prep.process(state)

      total_reward += reward

      if not tmp_skip and not skip:
        self.buffer.add({
            "state": tmp_state[0],
            "action": action,
            "reward": reward,
            "next_state": state[0],
            "done": int(done)
          })

      if not eval:
        if self.step >= self.steps_before_train and not self.solved and learn:
          # learn
          for _ in range(self.train_freq):
            self.train_step(self.buffer.sample(self.batch_size))
            _, self.step = self.session.run([self.inc_global_step, self.global_step])
        else:
          _, self.step = self.session.run([self.inc_global_step, self.global_step])

      if done:
        break

    if eval:
      tag = "eval_ep_reward"
    else:
      tag = "ep_reward"

    summary_value = summary_pb2.Summary.Value(tag=tag , simple_value=total_reward)
    summary_2 = summary_pb2.Summary(value=[summary_value])
    self.summary_writer.add_summary(summary_2, global_step=ep_idx)

    if self.max_reward is not None:
      self.solved = total_reward >= self.max_reward

    if self.step == self.num_steps:
      self.saver.save(self.session, self.summary_dir, global_step=self.step)

    return total_reward, self.step

  def close(self):
    self.session.close()