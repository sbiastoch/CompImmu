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
from sklearn.externals import joblib

features = [
	#PhysNumeric(),
	#Bin9(),
	Char6(),
	#SparseEncoding(),
	#PhysProperties()
	]


# Create a Data Model with various features
data_model = Data().loadFromFile(input_file).addFeatures(features)
# Save generated Features to a variable or to a file
data_model.formatCSV().save(output_file)
data = data_model.toArray()

# Give all the Data to the predictor and specify the training/validation/test ratios
pred = ScikitPredictor().setData(output_file).splitData(0.33)

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

# Save trained classifier 
pred.saveTrainedClassifier('/home/sbiastoch/Desktop/classifier.pkl')





'''
*** Example how a loaded classifier can used to predict sample peptides
'''

# Load previously trained classifier from disk 
loaded_pred = ScikitPredictor().loadTrainedClassifier('/home/sbiastoch/Desktop/classifier.pkl')

# Get the peptide sequence
peptids = ["LQKVPHTRY", "LAKVPHTRY", "AAAAAAAAA"]

# Populate Data-Object with peptides and add same Feature extractors as used when training the classifier
d = Data().loadFromArray(peptids).addFeatures(features)

# Get Features of peptides as arrays 
peptid_features = d.toFeatureArray()

# Classify the peptid represented by its features
print zip(peptids, loaded_pred.classify(peptid_features))