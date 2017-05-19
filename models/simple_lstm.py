import tensorflow as tf
import numpy as np
import data.utils as data_utils
import random

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def lstm_fn(features, targets, mode, params):
  print(features[0])
  print(targets[0])
  learning_rate = tf.train.exponential_decay(
      0.001,
      tf.contrib.framework.get_global_step(),
      10000,
      0.9,
      staircase=False)
  data_set = features

  random_number_01 = np.random.random_sample()
  bucket_id = min([
      i for i in xrange(len(train_buckets_scale))
      if train_buckets_scale[i] > random_number_01
  ])

  encoder_size, decoder_size = _buckets[bucket_id]
  encoder_inputs, decoder_inputs = [], []
  batch_encoder_inputs, batch_decoder_inputs = [], []

  for _ in xrange(batch_size):
    encoder_input, decoder_input = random.choice(data_set[bucket_id])

    encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
    encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

    decoder_pad_size = decoder_size - len(decoder_input) - 1
    decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                          [data_utils.PAD_ID] * decoder_pad_size)

  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y_pred, loss=cost, train_op=train_step)
