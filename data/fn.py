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


#https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/helpers.py
def batch(inputs, max_sequence_length=None):
  sequence_lengths = [len(seq) for seq in inputs]
  batch_size = len(inputs)

  if max_sequence_length is None:
    max_sequence_length = max(sequence_lengths)

  inputs_batch_major = np.zeros(
      shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

  for i, seq in enumerate(inputs):
    for j, element in enumerate(seq):
      inputs_batch_major[i, j] = element

  # [batch_size, max_time] -> [max_time, batch_size]
  inputs_time_major = inputs_batch_major.swapaxes(0, 1)

  return inputs_time_major, sequence_lengths


def lstm_input_fn(data_set, batch_size):
  print(data_set[0])
  input_features = tf.constant([i[0] for i in data_set])
  input_labels = tf.constant([i[1] for i in data_set])

  features, labels = tf.train.slice_input_producer(
      [input_features, input_labels])

  return tf.train.batch([features, labels], batch_size=batch_size)
