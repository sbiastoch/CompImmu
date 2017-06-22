'''
	Creates an sparse enconding for each amino acid, consisting of
	20 features per amino acid at position i. Thus, for a peptide 
	sequence of length 9, a total of 180 features are generated.
'''

from Bio.SubsMat import MatrixInfo

class BloDis:
    consense = None

    def __init__(self, consense):
        self.consense = consense;

    blosum = MatrixInfo.blosum62

    def _score_match(pair, matrix):
        if pair not in matrix:
            return matrix[(tuple(reversed(pair)))]
        else:
            return matrix[pair]

    def _switch(seq1):
        switch = []
        for i in range(len(seq1)):
            pair = (seq1[i], self.consense[i])
            switch += [score_match(pair, blosum)]
        return switch


    def getHeader(self):
        head = ['dist%s'%i for i in range(1,10)]
        return head

    def getFeatures(self, l):
        return switch(l['aa'])

