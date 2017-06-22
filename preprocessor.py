'''
	Main File of the Preprocessor
'''

# Add project folders to search path
import sys
sys.path.append('./feature_extractors')

# Include modules
from misc import Misc
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



###########################################
## Step 5: Evaluation                    ##
###########################################

#TODO: Test Trained Classifier with Test-Dataset



###########################################
## Step 6: Performance Visualization     ##
###########################################

#TODO: Print performance, draw graphs, ROC Curves etc.