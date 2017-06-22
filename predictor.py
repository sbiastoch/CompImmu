from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from paths import *

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = output_file
IRIS_TEST = output_file

def main():
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=2)
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

  # Fit model.
  classifier.fit(input_fn=get_train_inputs, steps=2000)

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  def new_samples():
  	return np.array(
      [[0.3174,0.5156,0.6747,0.8272,0.0,0.5,0.491,0.6048,0.7952,0.6914,0.0,0.5,0.6467,0.9852,0.2811,0.0,0.0,0.5,0.7964,0.8008,0.502,0.1605,0.6667,0.5,0.6467,0.9852,0.2811,0.0,0.0,0.5,0.3473,0.6771,0.5984,0.4568,0.0,0.5,0.485,0.9945,0.3655,0.1235,0.0,0.5,0.1737,0.3222,0.8514,0.5309,0.0,0.5,0.7725,0.8976,0.0763,0.037,0.6667,0.5],
	   [0.1737,0.3222,0.8514,0.5309,0.0,0.5,0.3473,0.6771,0.5984,0.4568,0.0,0.5,0.0,0.0,1.0,0.5062,0.0,0.5,0.1766,0.7679,0.8594,0.3827,0.0,0.5,0.6467,0.9852,0.2811,0.0,0.0,0.5,0.5569,0.5632,0.1124,0.679,0.5556,0.5,0.0,0.0,1.0,0.5062,0.0,0.5,0.3114,0.5506,0.2048,0.7441,0.0,0.5,0.6946,0.6738,0.6867,0.7901,0.0,1.0]], dtype=np.float32)

  predictions = list(classifier.predict(input_fn=new_samples))

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))

if __name__ == "__main__":
    main()