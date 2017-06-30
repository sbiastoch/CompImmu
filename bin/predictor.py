#!/usr/bin/env python
import argparse
import sys
sys.path.append('../')
sys.path.append('../feature_extractors')
from scikit_predictor import ScikitPredictor
from data import Data
from amino_acids import AminoAcids
from bin9 import Bin9
from blo_dist import BloDist
from char6 import Char6
from data import Data
from ic50 import IC50
from misc import Misc

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description = "Epitope Binding Predictor")
  parser.add_argument('input',metavar='in-file')
  parser.add_argument('output', metavar='out-file')
  args = parser.parse_args()

  with open(args.output, "w") as o:
    with open(args.input, "r") as i:
      for e in i:
        peptide = e.strip()

        # Load previously trained classifier from disk 
        pred = ScikitPredictor()
        features = pred.loadTrainedClassifier('best_classifier.pkl')

        # Populate Data-Object with peptides and add same Feature extractors as used when training the classifier
        d = Data().loadFromArray([peptide]).setFeatures(features)

        # Get Features of peptides as arrays 
        peptid_features = d.toFeatureArray()

        is_binder = pred.classify(peptid_features)

        o.write("%s\t%i\n"%(peptide,is_binder))

