from misc import Misc

class Char6:
	def getHeader(self):
		header = []
		for a in "A1 A2 A3 A4 A5 A6 A7 A8 A9".split(' '):
			for p in "hydrophobicity volume charge aromatic_side_chain hydrogen_bonds correction".split(' '): 
				header.append("char6_"+a+"-"+p)
		return header

	def getFeatures(self, l):
		features = []

		for a in l['aa']:
			features += map(lambda x: '10' if x == 'a' else x, Misc.knowledge_base[a]['char6'])

		return features