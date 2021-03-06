import tensorflow as tf


def simple_fn(features, targets, mode):
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))

  y = tf.matmul(features, W) + b

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=y))

  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
      cross_entropy, global_step=tf.contrib.framework.get_global_step())

  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y, loss=cross_entropy, train_op=train_step)
