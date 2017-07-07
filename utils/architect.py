import tensorflow as tf
from enum import Enum

def dense_block(input_node, layers, name, activation=tf.nn.relu, batch_norm_phase=None, last_layer_activation=False, detailed_summary=False):
  with tf.variable_scope(name):
    output = input_node
    for i, layer in enumerate(layers):
      if i == len(layers) - 1 and not last_layer_activation:
        output = tf.layers.dense(output, layer)
      else:
        output = tf.layers.dense(output, layer, activation=activation)
        if batch_norm_phase is not None:
          output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=batch_norm_phase)

      if detailed_summary:
        with tf.name_scope("layer_%d_output" % (i + 1)):
          variable_summaries(output)

  return output

def variable_summaries(var, name="summaries"):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def huber_loss(x, delta=1.0):
  return tf.where(
    tf.abs(x) < delta,
    tf.square(x) * 0.5,
    delta * (tf.abs(x) - 0.5 * delta)
  )

class NeuralNetwork():

  class Type(Enum):
    MLP = 1
    CNN_MLP = 2

  def __init__(self, config, type):
    self.config = config
    self.type = type

  def build(self, input_dim, output_dim, name):

    with tf.variable_scope(name):

      if self.type == self.Type.MLP:
          input_layer = tf.placeholder(tf.float32, shape=(None, input_dim))
          output_layer = dense_block(input_layer, [*self.config["hidden"], output_dim], "dense", batch_norm_phase=self.config["batch_norm"])
          return input_layer, output_layer
      elif self.type == self.Type.CNN_MLP:
        input_layer = tf.placeholder(tf.float32, shape=(None, input_dim[0], input_dim[1]))

