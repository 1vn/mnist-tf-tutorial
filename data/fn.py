import tensorflow as tf


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