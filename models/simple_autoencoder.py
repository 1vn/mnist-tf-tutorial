import tensorflow as tf

DROPOUT_PROB = 0.6


def autoencoder_fn(features, _, mode):

  n_hidden_1 = 50
  n_hidden_2 = 50
  n_hidden_3 = 50
  n_hidden_4 = 50
  n_hidden_5 = 50
  n_hidden_6 = 50
  n_hidden_7 = 50
  n_hidden_8 = 50
  n_hidden_9 = 50
  n_hidden_10 = 50
  n_input = 784

  weights = {
      'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
      'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
      'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
      'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
      'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
      'encoder_h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
      'encoder_h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
      'encoder_h8': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8])),
      'encoder_h9': tf.Variable(tf.random_normal([n_hidden_8, n_hidden_9])),
      'encoder_h10': tf.Variable(tf.random_normal([n_hidden_9, n_hidden_10])),
      'decoder_h1': tf.Variable(tf.random_normal([n_hidden_10, n_hidden_9])),
      'decoder_h2': tf.Variable(tf.random_normal([n_hidden_9, n_hidden_8])),
      'decoder_h3': tf.Variable(tf.random_normal([n_hidden_8, n_hidden_7])),
      'decoder_h4': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_6])),
      'decoder_h5': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_5])),
      'decoder_h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4])),
      'decoder_h7': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
      'decoder_h8': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
      'decoder_h9': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
      'decoder_h10': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
  }

  biases = {
      'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
      'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
      'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
      'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
      'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
      'encoder_b6': tf.Variable(tf.random_normal([n_hidden_6])),
      'encoder_b7': tf.Variable(tf.random_normal([n_hidden_7])),
      'encoder_b8': tf.Variable(tf.random_normal([n_hidden_8])),
      'encoder_b9': tf.Variable(tf.random_normal([n_hidden_9])),
      'encoder_b10': tf.Variable(tf.random_normal([n_hidden_10])),
      'decoder_b1': tf.Variable(tf.random_normal([n_hidden_9])),
      'decoder_b2': tf.Variable(tf.random_normal([n_hidden_8])),
      'decoder_b3': tf.Variable(tf.random_normal([n_hidden_7])),
      'decoder_b4': tf.Variable(tf.random_normal([n_hidden_6])),
      'decoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
      'decoder_b6': tf.Variable(tf.random_normal([n_hidden_4])),
      'decoder_b7': tf.Variable(tf.random_normal([n_hidden_3])),
      'decoder_b8': tf.Variable(tf.random_normal([n_hidden_2])),
      'decoder_b9': tf.Variable(tf.random_normal([n_hidden_1])),
      'decoder_b10': tf.Variable(tf.random_normal([n_input])),
  }

  def encoder(x):
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
    layer_5 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_4, weights['encoder_h5']), biases['encoder_b5']))
    layer_6 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_5, weights['encoder_h6']), biases['encoder_b6']))
    layer_7 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_6, weights['encoder_h7']), biases['encoder_b7']))
    layer_8 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_7, weights['encoder_h8']), biases['encoder_b8']))
    layer_9 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_8, weights['encoder_h9']), biases['encoder_b9']))
    layer_10 = tf.nn.sigmoid(
        tf.add(
            tf.matmul(layer_9, weights['encoder_h10']), biases['encoder_b10']))
    return layer_10

  def decoder(x):
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))
    layer_5 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_4, weights['decoder_h5']), biases['decoder_b5']))
    layer_6 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_5, weights['decoder_h6']), biases['decoder_b6']))
    layer_7 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_6, weights['decoder_h7']), biases['decoder_b7']))
    layer_8 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_7, weights['decoder_h8']), biases['decoder_b8']))
    layer_9 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_8, weights['decoder_h9']), biases['decoder_b9']))
    layer_10 = tf.nn.sigmoid(
        tf.add(
            tf.matmul(layer_9, weights['decoder_h10']), biases['decoder_b10']))
    return layer_10

  tf.summary.image("input", tf.reshape(features, [-1, 28, 28, 1]))

  encoder_op = encoder(features)
  dropout_op = tf.nn.dropout(encoder_op, DROPOUT_PROB)
  decoder_op = decoder(dropout_op)

  y_pred = decoder_op
  y_true = features

  tf.summary.image("output", tf.reshape(y_pred, [-1, 28, 28, 1]))

  # learning_rate = tf.train.exponential_decay(
  #     0.001,
  #     tf.contrib.framework.get_global_step(),
  #     10000,
  #     0.9,
  #     staircase=False)

  # cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + tf.nn.l2_loss(
  #     weights['encoder_h1']) + tf.nn.l2_loss(
  #         weights['encoder_h2']) + tf.nn.l2_loss(
  #             weights['decoder_h1']) + tf.nn.l2_loss(weights['decoder_h2'])
  cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

  train_step = tf.train.AdamOptimizer(0.001).minimize(
      cost, global_step=tf.contrib.framework.get_global_step())

  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y_pred, loss=cost, train_op=train_step)