'''
	Main File of the Preprocessor
'''

# Add project folders to search path
import sys
sys.path.append('./feature_extractors')

# Include modules
from misc import Misc
from tensorflow_predictor import TensorflowPredictor
from data import Data
from ic50 import IC50
from char6 import Char6
from bin9 import Bin9
from phys_properties import PhysProperties
from phys_numeric import PhysNumeric
from amino_acids import AminoAcids
from sparse_encoding import SparseEncoding
from paths import *
import tensorflow as tf

###########################################
## Step 1: Feature Extraction            ##
###########################################

data_model = Data(input_file).addFeatures([PhysNumeric])
data_model.formatTensorflowCSV().save(output_file)
#data_model.puts()
#data = data_model.toArray()




###########################################
## Step 2: Feature Selection             ##
###########################################

#TODO: Which Features of all the generated Features are usefull? -> PCA, etc...?
# May be omitted, so that Feature Selection takes only place by selecting the Feature Extractors



###########################################
## Step 3: Test-/Training set generation ##
###########################################

#TODO: Generate two independant datasets for Training and Testing



###########################################
## Step 4: Training                      ##
###########################################

#TODO: Train Classifier with Train-Dataset
pred = TensorflowPredictor().setTrainingData(output_file).setTestData(output_file)
pred.setClassifier().train(2000).evaluate()
unknown_samples = [[0.3174,0.5156,0.6747,0.8272,0.0,0.5,0.491,0.6048,0.7952,0.6914,0.0,0.5,0.6467,0.9852,0.2811,0.0,0.0,0.5,0.7964,0.8008,0.502,0.1605,0.6667,0.5,0.6467,0.9852,0.2811,0.0,0.0,0.5,0.3473,0.6771,0.5984,0.4568,0.0,0.5,0.485,0.9945,0.3655,0.1235,0.0,0.5,0.1737,0.3222,0.8514,0.5309,0.0,0.5,0.7725,0.8976,0.0763,0.037,0.6667,0.5],
	   [0.1737,0.3222,0.8514,0.5309,0.0,0.5,0.3473,0.6771,0.5984,0.4568,0.0,0.5,0.0,0.0,1.0,0.5062,0.0,0.5,0.1766,0.7679,0.8594,0.3827,0.0,0.5,0.6467,0.9852,0.2811,0.0,0.0,0.5,0.5569,0.5632,0.1124,0.679,0.5556,0.5,0.0,0.0,1.0,0.5062,0.0,0.5,0.3114,0.5506,0.2048,0.7441,0.0,0.5,0.6946,0.6738,0.6867,0.7901,0.0,1.0]]
pred.classify(unknown_samples)
###########################################
## Step 5: Evaluation                    ##
###########################################

#TODO: Test Trained Classifier with Test-Dataset



###########################################
## Step 6: Performance Visualization     ##
###########################################

#TODO: Print performance, draw graphs, ROC Curves etc.