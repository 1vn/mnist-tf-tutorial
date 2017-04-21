from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def model_fn(features, targets, mode):
	W = tf.Variable(tf.zeros([784, 10], dtype=tf.float64))
	b = tf.Variable(tf.zeros([10], dtype=tf.float64))

	y = tf.matmul(features, W) + b

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=y))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, 
		global_step=tf.contrib.framework.get_global_step())

	eval_metric_ops = { "accuracy": tf.metrics.accuracy(targets, y) }

	return tf.contrib.learn.ModelFnOps(
		mode=mode, 
		predictions=y, 
		loss=cross_entropy,
		train_op=train_step,
		eval_metric_ops=eval_metric_ops)



