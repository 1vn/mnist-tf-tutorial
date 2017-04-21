import tensorflow as tf


def input_fn(data_set, batch_size):
	x, y_ = data_set.next_batch(batch_size)
	return tf.convert_to_tensor(x, dtype=tf.float64), tf.convert_to_tensor(y_, dtype=tf.float64)
