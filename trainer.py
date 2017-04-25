
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

MODEL_DIR="tmp/training"

def accuracy_metric_fn(predictions, labels):
	correct_prediction = tf.equal(tf.argmax(labels,1), tf.argmax(predictions,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy

def main(_):	
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	est = tf.contrib.learn.Estimator(model_fn=model_fn, 
										model_dir=MODEL_DIR,
										config=tf.contrib.learn.RunConfig(save_checkpoints_secs=30))
	exp = tf.contrib.learn.Experiment(estimator=est, 
										eval_steps=250,
										min_eval_frequency=500,
										local_eval_frequency=500,
										train_input_fn=lambda: input_fn(mnist.train, 100), 
										eval_input_fn=lambda: input_fn(mnist.test, 100),
										eval_metrics={"accuracy": accuracy_metric_fn})

	exp.train_and_evaluate()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
	                  help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)