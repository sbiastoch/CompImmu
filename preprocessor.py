'''
	Main File of the Preprocessor
'''

# Import all files
from misc import Misc
from data import Data
from ic50 import IC50
from amino_acids import AminoAcids
from sparse_encoding import SparseEncoding
from paths import *

data = Data(input_file).\
	addFeatures([IC50, AminoAcids, SparseEncoding]).\
	formatCSV().\
	save(output_file).\
	puts().\
	toArray()

print data