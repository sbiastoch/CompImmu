from misc import Misc

class PhysNumeric:
	props = "volume bulkiness flexibility plarity aromaticity charge".split(' ')

	def getHeader(self):
		header = []
		for a in "A1 A2 A3 A4 A5 A6 A7 A8 A9".split(' '):
			for p in self.props: 
				header.append("physNum_"+a+"-"+str(p))
		return header

	def getFeatures(self, l):
		features = []

		for a in l['aa']:
			features += [ str(Misc.knowledge_base[a][p]) for p in self.props ]

		return features