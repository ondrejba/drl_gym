import tensorflow as tf

def dense_block(input_node, layers, name, activation=tf.nn.relu, last_layer_activation=False, detailed_summary=False):

  with tf.variable_scope(name):
    output = input_node
    for i, layer in enumerate(layers):
      if i == len(layers) - 1 and not last_layer_activation:
        output = tf.layers.dense(output, layer)
      else:
        output = tf.layers.dense(output, layer, activation=activation)

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