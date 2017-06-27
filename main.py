'''
	Main File of the Preprocessor
'''

# Add project folders to search path

# Include modules

import sys
sys.path.append('./feature_extractors')
from amino_acids import AminoAcids
from bin9 import Bin9
from blo_dis import BloDis
from char6 import Char6
from data import Data
from ic50 import IC50
from misc import Misc
from paths import *
from phys_numeric import PhysNumeric
from phys_properties import PhysProperties
from scikit_predictor import ScikitPredictor
from sklearn.neural_network import MLPClassifier
from sparse_encoding import SparseEncoding
from tensorflow_predictor import TensorflowPredictor
import numpy as np
import tensorflow as tf




# Create a Data Model with various features
data_model = Data(input_file).addFeatures([
	#PhysNumeric(),
	#Bin9(),
	Char6(),
	#SparseEncoding(),
	#PhysProperties()
	])
# Save generated Features to a variable or to a file
data_model.formatCSV().save(output_file)
#data = data_model.toArray()

# Give all the Data to the predictor and specify the training/validation/test ratios
pred = ScikitPredictor().setData(output_file).splitData(0.2, 0.2)

# Build a classifier
base_classifier = MLPClassifier(max_iter=1000)

# Give the classifier to the predictor
pred.setClassifier(base_classifier)

best_classifier = pred.optimize()
pred.setClassifier(best_classifier)

# Example Cross-Validation run on a 5-Fold split. Crossvalidation is performed on the validation-dataset
# pred.crossValidate(5)

# Train the classifier on training-dataset for n iterations
pred.train(5000)

# Do the final evaluation on the previously unseen test-data
pred.evaluateOnTestData()

# Print ROC-curve
pred.plot_roc()