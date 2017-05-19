import numpy as np
import tensorflow as tf
import data.utils as data_utils
import random

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def input_fn(data_set, batch_size):
  input_images = tf.constant(data_set.images)
  input_labels = tf.constant(data_set.labels)

  image, label = tf.train.slice_input_producer([input_images, input_labels])
  return tf.train.batch([image, label], batch_size=batch_size)


def autoencoder_input_fn(data_set, batch_size):
  input_images = tf.constant(data_set.images)
  input_labels = tf.constant(data_set.labels)

  image, label = tf.train.slice_input_producer([input_images, input_labels])
  return tf.train.batch([image, image], batch_size=batch_size)


def lstm_input_fn(data_set, batch_size, train_buckets_scale):
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

  return tf.train.batch([encoder_inputs, decoder_inputs], batch_size=batch_size)
