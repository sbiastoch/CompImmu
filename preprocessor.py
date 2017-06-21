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

#########################

data = Data(input_file).\
	addFeatures([AminoAcids, SparseEncoding, Bin9, PhysNumeric, PhysProperties, Char6, IC50]).\
	formatCSV().\
	save(output_file).\
	puts().\
	toArray()

print data