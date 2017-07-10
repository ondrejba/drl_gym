import tensorflow as tf
import math, os
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.core.framework import summary_pb2

import utils.utils as utils
import utils.architect as architect
from utils.ReplayBuffer import ReplayBuffer

class DDPG:

  CRITIC_NAME = "critic"
  TARGET_CRITIC_NAME = "target_critic"

  ACTOR_NAME = "actor"
  TARGET_ACTOR_NAME = "target_actor"

  def __init__(self, prep, policy, state_dim, action_dim, action_high, action_low, monitor_directory, actor_learning_rate=1e-2, critic_learning_rate=1e-3,
               critic_target_update_rate=1e-3, actor_target_update_rate=1e-3, discount=0.99,
               l2_decay=1e-2, buffer_size=10000, steps_before_train=1000, max_reward=None, train_freq=1,
               num_steps=500000, batch_size=64):

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
    self.decay_biases = False
    self.steps_before_train = steps_before_train
    self.max_reward = max_reward
    self.train_freq = train_freq
    self.num_steps = num_steps
    self.summary_dir = os.path.join(monitor_directory, "summary")

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

    with tf.variable_scope(name):

      # weights and biases
      W1 = tf.get_variable("W1", (self.state_dim, 400), initializer=initializers.xavier_initializer())
      b1 = tf.get_variable("b1", 400, initializer=tf.zeros_initializer())

      W2 = tf.get_variable("W2", (400, 300), initializer=initializers.xavier_initializer())
      b2 = tf.get_variable("b2", 300, initializer=tf.zeros_initializer())

      W2_action = tf.get_variable("W2_action", (self.action_dim, 300), initializer=initializers.xavier_initializer())

      W3 = tf.get_variable("W3", (300, 1), initializer=initializers.xavier_initializer())
      b3 = tf.get_variable("b3", 1, initializer=tf.zeros_initializer())

      # layers
      layer_1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
      layer_2 = tf.nn.relu(tf.matmul(layer_1, W2) + tf.matmul(action_input, W2_action) + b2)
      output_layer = tf.matmul(layer_2, W3) + b3

      # weight decay
      weights = [W1, W2, W2_action, W3]
      if self.decay_biases:
        weights += [b1, b2, b3]

      weight_decay = tf.add_n([self.l2_decay * tf.nn.l2_loss(var) for var in weights])

      return output_layer, weight_decay

  def build_actor(self, name, state_input):

    with tf.variable_scope(name):

      # weights and biases
      W1 = tf.get_variable("W1", (self.state_dim, 400), initializer=initializers.xavier_initializer())
      b1 = tf.get_variable("b1", 400, initializer=tf.zeros_initializer())

      W2 = tf.get_variable("W2", (400, 300), initializer=initializers.xavier_initializer())
      b2 = tf.get_variable("b2", 300, initializer=tf.zeros_initializer())

      W3 = tf.get_variable("W3", (300, 1), initializer=initializers.xavier_initializer())
      b3 = tf.get_variable("b3", 1, initializer=tf.zeros_initializer())

      # layers
      layer_1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
      layer_2 = tf.nn.relu(tf.matmul(layer_1, W2) + b2)

      output_layer = tf.matmul(layer_2, W3) + b3

      if self.action_high == math.inf and self.action_low == - math.inf:
        return output_layer
      elif abs(self.action_high) == abs(self.action_low):
        return tf.nn.tanh(output_layer) * self.action_high
      else:
        raise ValueError("Asymmetric action range.")

  def build(self):

    self.state_input = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="state_input")
    self.next_state_input = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="next_state_input")
    self.action_input = tf.placeholder(tf.float32, shape=(None, self.action_dim), name="action_input")
    self.reward_input = tf.placeholder(tf.float32, shape=(None,), name="reward_input")
    self.done_input = tf.placeholder(tf.float32, shape=(None,), name="done_input")

    self.target_action = self.build_actor(self.TARGET_ACTOR_NAME, self.next_state_input)

    self.q_value, weight_decay = self.build_critic(self.CRITIC_NAME, self.state_input, self.action_input)
    self.target_q_value, _ = self.build_critic(self.TARGET_CRITIC_NAME, self.next_state_input, self.target_action)

    self.targets = tf.expand_dims(self.reward_input, 1) + self.discount * (1 - tf.expand_dims(self.done_input, 1)) * self.target_q_value
    self.loss = tf.reduce_mean(tf.square(tf.stop_gradient(self.targets) - self.q_value)) + weight_decay
    self.critic_train_op = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.loss)
    self.action_gradients = tf.gradients(self.q_value, self.action_input)[0]

    tf.summary.scalar("critic_loss", self.loss)

    self.action = self.build_actor(self.ACTOR_NAME, self.state_input)

    trainable_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ACTOR_NAME)
    #self.q_gradient_input = tf.placeholder(tf.float32, (None, self.action_dim))
    self.actor_params_gradient = tf.gradients(self.action, trainable_weights, - self.action_gradients)
    self.actor_train_op = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(zip(self.actor_params_gradient, trainable_weights))

    self.target_critic_update = architect.create_target_update_ops(self.CRITIC_NAME, self.TARGET_CRITIC_NAME, self.critic_target_update_rate)
    self.target_actor_update = architect.create_target_update_ops(self.ACTOR_NAME, self.TARGET_ACTOR_NAME, self.actor_target_update_rate)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.inc_global_step = tf.assign(self.global_step, tf.add(self.global_step, 1))

  def train_actor(self, states):
    actions = self.session.run(self.action, feed_dict={
      self.state_input: states
    })

    self.session.run(self.actor_train_op, feed_dict={
      self.state_input: states,
      self.action_input: actions
    })

  def train_critic(self, states, actions, rewards, next_states, done):
    _, summary = self.session.run([self.critic_train_op, self.merged], feed_dict={
      self.state_input: states,
      self.action_input: actions,
      self.reward_input: rewards,
      self.next_state_input: next_states,
      self.done_input: done
    })
    self.summary_writer.add_summary(summary, global_step=self.step)

  def train_step(self, batch):
    self.train_critic(batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["done"])
    self.train_actor(batch["states"])
    self.session.run([self.target_critic_update, self.target_actor_update])

  def run_episode(self, env):

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
          self.state_input: state
        })[0]
        action = self.policy.add_noise(action)

      tmp_state = state
      tmp_skip = skip

      state, reward, done, _ = env.step(action)
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

      if self.step >= self.steps_before_train and not self.solved:
        # learn
        for _ in range(self.train_freq):
          self.train_step(self.buffer.sample(self.batch_size))
          _, self.step = self.session.run([self.inc_global_step, self.global_step])
      else:
        _, self.step = self.session.run([self.inc_global_step, self.global_step])

      if done:
        break

    summary_value = summary_pb2.Summary.Value(tag="episode_reward", simple_value=total_reward)
    summary_2 = summary_pb2.Summary(value=[summary_value])
    self.summary_writer.add_summary(summary_2, global_step=self.step)

    if self.max_reward is not None:
      self.solved = total_reward >= self.max_reward

    if self.step == self.num_steps:
      self.saver.save(self.session, self.summary_dir, global_step=self.step)

    return total_reward, self.step