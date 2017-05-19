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
import data.utils as data_utils

import tensorflow as tf

MODEL_DIR = "tmp/training"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'simple', 'simple or autoencoder or lstm.')

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


#https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/translate.py
def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


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

  train_input_fn = None
  eval_input_fn = None

  if FLAGS.model == "autoencoder":
    model_fn = autoencoder_fn
    selected_input_fn = autoencoder_input_fn
    eval_metrics["euclidean distance"] = tf.contrib.learn.MetricSpec(
        metric_fn=euclidean_distance_metric)
  elif FLAGS.model == "lstm":
    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
        "LANG_data/",
        4000,
        4000,)
    selected_input_fn = lstm_input_fn
    dev_set = read_data(from_dev, to_dev)
    train_set = read_data(
        from_train,
        to_train,)

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    train_buckets_scale = [
        sum(train_bucket_sizes[:i + 1]) / train_total_size
        for i in xrange(len(train_bucket_sizes))
    ]
    train_input_fn = lambda: selected_input_fn(train_set, 100, train_buckets_scale)
    eval_input_fn = lambda: selected_input_fn(dev_set, 100, train_buckets_scale)

    model_fn = lstm_fn

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