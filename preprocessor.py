# Import all files
from misc import Misc
from data import Data
from c50 import C50
from amino_acids import AminoAcids
from sparse_encoding import SparseEncoding
from paths import *

Data(input_file).addFeatures([C50, AminoAcids, SparseEncoding]).\
		   formatCSV().\
		   save(output_file).\
		   puts()