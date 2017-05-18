from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants

from models.simple_mlp import simple_fn
from models.simple_autoencoder import autoencoder_fn
from models.simple_lstm import lstm_fn

from data.fn import input_fn, autoencoder_input_fn, lstm_input_fn

import tensorflow as tf

MODEL_DIR = "tmp/training"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'simple', 'simple or autoencoder or lstm.')


def accuracy_metric_fn(predictions, labels):
  correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy


def euclidean_distance_metric(predictions, labels):
  d = tf.reduce_mean(
      tf.sqrt(
          tf.reduce_sum(
              tf.square(tf.subtract(predictions, tf.cast(labels, tf.float32))),
              1)))
  return d


def main(_):

  print("running with model: ", FLAGS.model)

  model_fn = simple_fn
  selected_input_fn = input_fn
  eval_metrics = {
      "accuracy": tf.contrib.learn.MetricSpec(metric_fn=accuracy_metric_fn)
  }

  if FLAGS.model == "autoencoder":
    model_fn = autoencoder_fn
    selected_input_fn = autoencoder_input_fn
    eval_metrics["euclidean distance"] = tf.contrib.learn.MetricSpec(
        metric_fn=euclidean_distance_metric)
  elif FLAGS.model == "lstm":
    model_fn = lstm_fn
    selected_input_fn = lstm_input_fn

  if (FLAGS.model == "simple" or FLAGS.model == "autoencoder"):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_input_fn = lambda: selected_input_fn(mnist.train, 100)
    eval_input_fn = lambda: selected_input_fn(mnist.test, 100)

  est = tf.contrib.learn.Estimator(
      model_fn=model_fn,
      model_dir=MODEL_DIR,
      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=30))
  exp = tf.contrib.learn.Experiment(
      estimator=est,
      eval_steps=1,
      min_eval_frequency=1,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      eval_metrics=eval_metrics)
  exp.train_and_evaluate()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()