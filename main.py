# Add project folders to search path
import sys
sys.path.append('./feature_extractors')

# Import of project files
from amino_acids import AminoAcids
from bin9 import Bin9
from blo_dist import BloDist
from char6 import Char6
from data import Data
from ic50 import IC50
from misc import Misc
from paths import *
from phys_numeric import PhysNumeric
from phys_properties import PhysProperties
from scikit_predictor import ScikitPredictor
from sparse_encoding import SparseEncoding

# Import of libraries
from sklearn.neural_network import MLPClassifier

features = [
	PhysNumeric(),
	Bin9(),
	Char6(),
	SparseEncoding(),
	PhysProperties(),
	BloDist('AQIDNYNKF')
]

data_model = Data().loadFromFile(input_file)

for feature in features:

	feature_name = feature.__class__.__name__

	# Create a Data Model with various features
	data_model.setFeatures([feature])

	# Save generated Features to a variable or to a file
	data_model.formatCSV().save(output_file)

	# Give all the Data to the predictor and specify the training/validation/test ratios
	pred = ScikitPredictor().setData(output_file).splitData(0.33)

	# Build a classifier
	base_classifier = MLPClassifier(max_iter=500)

	# Give the classifier to the predictor
	pred.setClassifier(base_classifier)

	best_classifier = pred.optimize()
	pred.setClassifier(best_classifier)

	# Train the classifier on training-dataset for n iterations
	pred.train(5000)

	# Do the final evaluation on the previously unseen test-data
	pred.evaluateOnTestData()

	# Save trained classifier 

	all_params = pred.classifier.get_params()
	selected_params = ['learning_rate_init', 'momentum', 'solver', 'hidden_layer_sizes', 'activation']
   	params = {k + "_" + str(all_params[k]) for k in selected_params} 
	filename = feature_name+'__' + "-".join(params)
	pred.saveTrainedClassifier(feature, 'trained_classifiers/'+filename+'.pkl')
	
	# Print ROC-curve
	pred.plot_roc(feature_name, filename)




'''
*** Example how a loaded classifier can used to predict sample peptides
'''
'''
# Load previously trained classifier from disk 
loaded_pred = ScikitPredictor()
feature_extractor = loaded_pred.loadTrainedClassifier('trained_classifiers/'+filename+'.pkl')

# Get the peptide sequence
peptids = ["LQKVPHTRY", "LAKVPHTRY", "AAAAAAAAA"]

# Populate Data-Object with peptides and add same Feature extractors as used when training the classifier
d = Data().loadFromArray(peptids).setFeatures(feature_extractor)

# Get Features of peptides as arrays 
peptid_features = d.toFeatureArray()

# Classify the peptid represented by its features
print zip(peptids, pred.classify(peptid_features))
'''