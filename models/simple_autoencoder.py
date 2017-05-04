import tensorflow as tf


def autoencoder_fn(features, _, mode):
  n_hidden_1 = 256
  n_hidden_2 = 128
  n_input = 784

  weights = {
      'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
      'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
      'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
      'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
  }

  biases = {
      'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
      'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
      'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
      'decoder_b2': tf.Variable(tf.random_normal([n_input])),
  }

  def encoder(x):
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

  def decoder(x):
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

  tf.summary.image("input", tf.reshape(features, [-1, 28, 28, 1]))

  encoder_op = encoder(features)
  decoder_op = decoder(encoder_op)

  y_pred = decoder_op
  y_true = features

  tf.summary.image("output", tf.reshape(y_pred, [-1, 28, 28, 1]))

  learning_rate = tf.train.exponential_decay(
      0.5, tf.contrib.framework.get_global_step(), 25000, 0.9, staircase=True)

  cost = tf.nn.l2_loss(tf.reduce_mean(tf.pow(y_true - y_pred, 2)))
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
      cost, global_step=tf.contrib.framework.get_global_step())

  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y_pred, loss=cost, train_op=train_step)