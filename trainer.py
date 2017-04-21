
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants

from models.fn import model_fn
from data.fn import input_fn

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
	tf.contrib.framework.get_or_create_global_step(graph=None)

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	est = tf.contrib.learn.Estimator(model_fn=model_fn)
	exp = tf.contrib.learn.Experiment(estimator=est, 
										train_input_fn=lambda: input_fn(mnist.train, 100), 
										eval_input_fn=input_fn(mnist.test, 100))

	exp.train()

	exp.evaluate()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
	                  help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)