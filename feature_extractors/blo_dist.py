from Bio.SubsMat import MatrixInfo

class BloDist:
    consense = None

    def __init__(self, consense):
        self.consense = consense;

    blosum = MatrixInfo.blosum62

    def _score_match(self, pair, matrix):
        if pair not in matrix:
            return matrix[(tuple(reversed(pair)))]
        else:
            return matrix[pair]

    def _switch(self, seq1):
        switch = []
        for i in range(len(seq1)):
            pair = (seq1[i], self.consense[i])
            switch += [str(self._score_match(pair, self.blosum))]
        return switch


    def getHeader(self):
        head = ['dist%s'%i for i in range(1,10)]
        return head

    def getFeatures(self, l):
        return self._switch(l['aa'])

