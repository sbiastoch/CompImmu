'''
	Creates an sparse enconding for each amino acid, consisting of
	20 features per amino acid at position i. Thus, for a peptide 
	sequence of length 9, a total of 180 features are generated.
'''

from misc import Misc

class Bin9:
	def getHeader(self):
		header = []
		for a in "A1 A2 A3 A4 A5 A6 A7 A8 A9".split(' '):
			for p in range(1, 10): 
				header.append("bin9_"+a+"-prop"+str(p))
		return header

	def getFeatures(self, l):
		features = []

		for a in l['aa']:
			features += list(Misc.knowledge_base[a]['bin9'])

		return features