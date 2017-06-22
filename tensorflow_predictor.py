from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from paths import *

import os
import urllib

import numpy as np
import tensorflow as tf

class TensorflowPredictor:

	# Data sets
	training_set = None
	test_set = None

	classifier = None


	def setTrainingData(self, csv_path):
	  # Load datasets.
	  self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
	      filename=csv_path,
	      target_dtype=np.int,
	      features_dtype=np.float32)
	  return self


  	def setTestData(self, csv_path):
	  self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
	      filename=csv_path,
	      target_dtype=np.int,
	      features_dtype=np.float32)
	  return self


	def setClassifier(self):
	  # Specify that all features have real-value data
	  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]

	  # Build 3 layer DNN with 10, 20, 10 units respectively.
	  self.classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
	                                              hidden_units=[10, 20, 10],
	                                              n_classes=2)
	  return self

	def train(self, iterations):
	  # Define the training inputs
	  def get_train_inputs():
	    x = tf.constant(self.training_set.data)
	    y = tf.constant(self.training_set.target)

	    return x, y

	  # Fit model.
	  self.classifier.fit(input_fn=get_train_inputs, steps=iterations)
	  return self

	def evaluate(self):

	  # Define the test inputs
	  def get_test_inputs():
	    x = tf.constant(self.test_set.data)
	    y = tf.constant(self.test_set.target)
	    return x, y

	  # Evaluate accuracy.
	  accuracy_score = self.classifier.evaluate(input_fn=get_test_inputs,
	                                       steps=1)["accuracy"]

	  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
	  return self

	def classify(self, samples):
	  new_samples = lambda: np.array(samples, dtype=np.float32)
	  predictions = list(self.classifier.predict(input_fn=new_samples))
	  print("New Samples, Class Predictions:    {}\n".format(predictions))
	  return self