import tensorflow as tf
import math, os
from tensorflow.core.framework import summary_pb2

import utils.utils as utils
import utils.architect as architect
from utils.ReplayBuffer import ReplayBuffer

class DDPG:

  CRITIC_NAME = "critic"
  TARGET_CRITIC_NAME = "target_critic"

  ACTOR_NAME = "actor"
  TARGET_ACTOR_NAME = "target_actor"

  def __init__(self, state_dim, action_dim, monitor_directory, actor_learning_rate=1e-5, critic_learning_rate=1e-3,
               critic_target_update_rate=1e-3, actor_target_update_rate=1e-3, discount=0.99, l2_decay=1e-2,
               buffer_size=1000000, batch_size=64, detail_summary=False, tanh_action=True, input_batch_norm=True,
               all_batch_norm=True, log_frequency=10):

    self.state_dim = state_dim
    self.action_dim = action_dim

    self.critic_learning_rate = critic_learning_rate
    self.actor_learning_rate = actor_learning_rate
    self.critic_target_update_rate = critic_target_update_rate
    self.actor_target_update_rate = actor_target_update_rate
    self.discount = discount
    self.batch_size = batch_size
    self.l2_decay = l2_decay
    self.buffer_size = buffer_size
    self.summary_dir = os.path.join(monitor_directory, "summary")
    self.detail_summary = detail_summary
    self.tanh_action = tanh_action
    self.input_batch_norm = input_batch_norm
    self.all_batch_norm = all_batch_norm
    self.log_frequency = log_frequency

    self.step = 0
    self.solved = False

    self.buffer = ReplayBuffer(buffer_size, self.state_dim, self.action_dim)

    self.__build()

    self.summary_dir = utils.new_summary_dir(self.summary_dir)
    utils.log_params(self.summary_dir, {
      "actor learning rate": self.actor_learning_rate,
      "critic learning rate": self.critic_learning_rate,
      "batch size": self.batch_size,
      "actor update rate": self.actor_target_update_rate,
      "critic update rate": self.critic_target_update_rate,
      "buffer size": self.buffer_size,
    })

    self.saver = tf.train.Saver(max_to_keep=None)

    init_op = tf.global_variables_initializer()
    self.session = tf.Session()

    self.merged = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

    self.session.run(init_op)

  """
  PUBLIC
  """

  def learn(self):

    batch = self.buffer.sample(self.batch_size)
    self.__train_critic(batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["done"])
    self.__train_actor(batch["states"])

    self.session.run([self.target_critic_update, self.target_actor_update, self.inc_global_step])

  def act(self, state):
    a =  self.session.run(self.action, feed_dict={
      self.state_input: state,
      self.is_training: False
    })[0]
    return a

  def perceive(self, transition):
    self.buffer.add(transition)

  def log_scalar(self, name, value, index):
    summary_value = summary_pb2.Summary.Value(tag=name, simple_value=value)
    summary_2 = summary_pb2.Summary(value=[summary_value])
    self.summary_writer.add_summary(summary_2, global_step=index)

  def save(self):
    self.saver.save(self.session, self.summary_dir, global_step=self.session.run(self.global_step))

  def close(self):
    self.session.close()

  """
  PRIVATE
  """

  def __build_critic(self, name, state_input, action_input):

    bn_training = self.is_training
    if name == self.TARGET_CRITIC_NAME:
      bn_training = False

    with tf.variable_scope(name):

      # weights and biases
      W1 = self.__get_weights((self.state_dim, 400), self.state_dim, name="W1")
      b1 = self.__get_weights((400,), self.state_dim, name="b1")

      W2 = self.__get_weights((400, 300), 400 + self.action_dim, name="W2")
      b2 = self.__get_weights((300,), 400 + self.action_dim, name="b2")

      W2_action = self.__get_weights((self.action_dim, 300), 400 + self.action_dim, name="W2_action")

      W3 = tf.Variable(tf.random_uniform((300, 1), -3e-3, 3e-3), name="W3")
      b3 = tf.Variable(tf.random_uniform((1,), -3e-3, 3e-3), name="b3")

      # layers
      if self.input_batch_norm:
        state_input = tf.layers.batch_normalization(state_input, training=bn_training)

      layer_1 = tf.matmul(state_input, W1) + b1

      if self.all_batch_norm:
        layer_1 = tf.layers.batch_normalization(layer_1, training=bn_training)

      layer_1 = tf.nn.relu(layer_1)

      layer_2 = tf.nn.relu(tf.matmul(layer_1, W2) + tf.matmul(action_input, W2_action) + b2)

      output_layer = tf.matmul(layer_2, W3) + b3

      # summary
      if name == self.CRITIC_NAME:
        self.critic_summaries = [
          tf.summary.histogram("W1", W1),
          tf.summary.histogram("b1", b1),

          tf.summary.histogram("W2", W2),
          tf.summary.histogram("b2", b2),
          tf.summary.histogram("W2_action", W2_action),

          tf.summary.histogram("W3", W3),
          tf.summary.histogram("b3", b3),

          tf.summary.histogram("layer_1", layer_1),
          tf.summary.histogram("layer_2", layer_2),
          tf.summary.histogram("output_layer", output_layer)
        ]

      # weight decay
      weights = [W1, b1, W2, b2, W2_action, W3, b3]
      weight_decay = tf.add_n([self.l2_decay * tf.nn.l2_loss(var) for var in weights])

      return output_layer, weight_decay

  def __build_actor(self, name, state_input):

    bn_training = self.is_training
    if name == self.TARGET_ACTOR_NAME:
      bn_training = False

    with tf.variable_scope(name):

      # weights and biases
      W1 = self.__get_weights((self.state_dim, 400), self.state_dim, name="W1")
      b1 = self.__get_weights((400,), self.state_dim, name="b1")

      W2 = self.__get_weights((400, 300), 400, name="W2")
      b2 = self.__get_weights((300,), 400, name="b2")

      W3 = tf.Variable(tf.random_uniform((300, self.action_dim), minval=-3e-3, maxval=3e-3), name="W3")
      b3 = tf.Variable(tf.random_uniform((self.action_dim,), -3e-3, 3e-3), name="b3")

      # layers
      if self.input_batch_norm:
        state_input = tf.layers.batch_normalization(state_input, training=bn_training)

      layer_1 = tf.matmul(state_input, W1) + b1

      if self.all_batch_norm:
        layer_1 = tf.layers.batch_normalization(layer_1, training=bn_training)

      layer_1 = tf.nn.relu(layer_1)

      layer_2 = tf.matmul(layer_1, W2) + b2

      if self.all_batch_norm:
        layer_2 = tf.layers.batch_normalization(layer_2, training=bn_training)

      layer_2 = tf.nn.relu(layer_2)

      output_layer = tf.matmul(layer_2, W3) + b3

      # summary
      if name == self.ACTOR_NAME:
        self.actor_summaries = [
          tf.summary.histogram("W1", W1),
          tf.summary.histogram("b1", b1),

          tf.summary.histogram("W2", W2),
          tf.summary.histogram("b2", b2),

          tf.summary.histogram("W3", W3),
          tf.summary.histogram("b3", b3),

          tf.summary.histogram("layer_1", layer_1),
          tf.summary.histogram("layer_2", layer_2),
          tf.summary.histogram("output_layer", output_layer)
        ]

      if self.tanh_action:
        return tf.nn.tanh(output_layer)
      else:
        return output_layer

  def __build(self):

    self.state_input = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="state_input")
    self.next_state_input = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="next_state_input")
    self.action_input = tf.placeholder(tf.float32, shape=(None, self.action_dim), name="action_input")
    self.reward_input = tf.placeholder(tf.float32, shape=(None,), name="reward_input")
    self.done_input = tf.placeholder(tf.float32, shape=(None,), name="done_input")
    self.is_training = tf.placeholder(tf.bool, name="is_training")

    # inputs summary
    if self.detail_summary:
      self.input_summaries = [
        tf.summary.histogram("state", self.state_input),
        tf.summary.histogram("next_state", self.next_state_input),
        tf.summary.histogram("action", self.action_input),
        tf.summary.histogram("reward", self.reward_input),
        tf.summary.histogram("done", self.done_input)
      ]

    self.target_action = self.__build_actor(self.TARGET_ACTOR_NAME, self.next_state_input)

    self.q_value, weight_decay = self.__build_critic(self.CRITIC_NAME, self.state_input, self.action_input)
    self.target_q_value, _ = self.__build_critic(self.TARGET_CRITIC_NAME, self.next_state_input, self.target_action)

    self.tmp = tf.expand_dims(self.reward_input, 1)

    self.targets = tf.expand_dims(self.reward_input, 1) + self.discount * (1 - tf.expand_dims(self.done_input, 1)) * self.target_q_value
    self.diff = self.targets - self.q_value

    self.loss = tf.reduce_mean(tf.square(tf.stop_gradient(self.targets) - self.q_value)) + weight_decay
    self.loss_summary = tf.summary.scalar("critic_loss", self.loss)

    self.critic_train_op = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.loss)

    # add critic batch norm. update
    if self.input_batch_norm or self.all_batch_norm:
      self.critic_bn_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.CRITIC_NAME)
      self.critic_bn_update_op = tf.group(*self.critic_bn_update_op)
      self.critic_train_op = tf.group(self.critic_train_op, self.critic_bn_update_op)

    self.action = self.__build_actor(self.ACTOR_NAME, self.state_input)
    self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ACTOR_NAME)
    self.action_gradients = tf.gradients(self.q_value, self.action_input)[0]
    self.actor_params_gradient = tf.gradients(self.action, self.actor_params, - self.action_gradients)

    # actor gradients summary
    if self.detail_summary:
      self.actor_summaries.append(tf.summary.histogram("action_gradient", self.action_gradients))
      for grad in self.actor_params_gradient:
        self.actor_summaries.append(tf.summary.histogram("actor_parameter_gradients", grad))

    self.actor_train_op = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(zip(self.actor_params_gradient, self.actor_params))

    # add actor batch norm. update
    if self.input_batch_norm or self.all_batch_norm:
      self.actor_bn_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.ACTOR_NAME)
      self.actor_bn_update_op = tf.group(*self.actor_bn_update_op)
      self.actor_train_op = tf.group(self.actor_train_op, self.actor_bn_update_op)

    self.target_critic_update = architect.create_target_update_ops(self.CRITIC_NAME, self.TARGET_CRITIC_NAME, self.critic_target_update_rate)
    self.target_actor_update = architect.create_target_update_ops(self.ACTOR_NAME, self.TARGET_ACTOR_NAME, self.actor_target_update_rate)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.inc_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1))

    # group summaries
    self.critic_summaries = tf.summary.merge(self.critic_summaries)

    if self.detail_summary:
      self.actor_summaries = tf.summary.merge(self.actor_summaries)
      self.input_summaries = tf.summary.merge(self.input_summaries)

  @staticmethod
  def __get_weights(shape, input_shape, name="var"):
    return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(input_shape), 1 / math.sqrt(input_shape)), name=name)

  def __train_actor(self, states):

    actions = self.session.run(self.action, feed_dict={
      self.state_input: states,
      self.is_training: True
    })

    self.session.run(self.actor_train_op, feed_dict={
      self.state_input: states,
      self.action_input: actions,
      self.is_training: True
    })

  def __train_critic(self, states, actions, rewards, next_states, done):
    feed_dict = {
      self.state_input: states,
      self.action_input: actions,
      self.reward_input: rewards,
      self.next_state_input: next_states,
      self.done_input: done,
      self.is_training: True
    }
    step = self.session.run(self.global_step)

    if step % self.log_frequency == 0:

      ops = [self.critic_train_op, self.loss_summary]

      if self.detail_summary:
        ops.append(self.actor_summaries)
        ops.append(self.input_summaries)

      res = self.session.run(ops, feed_dict=feed_dict)

      self.summary_writer.add_summary(res[1], global_step=step)

      if self.detail_summary:
        self.summary_writer.add_summary(res[2], global_step=step)
        self.summary_writer.add_summary(res[3], global_step=step)
    else:
      self.session.run(self.critic_train_op, feed_dict=feed_dict)