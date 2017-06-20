from misc import Misc

class SparseEncoding:
	def getHeader(self):
		header = []
		for aaPosition in range(0, 9):
			for posibleAA in Misc.aas:
				header.append("AA%s-"%(aaPosition) + posibleAA)
		return header

	def getFeatures(self, l):
		sparseEncoding = []
		for i in range(0, 9):
			a = l['aa'][i]
			tmp = ['0']*20
			tmp[Misc.aas.index(a)] = '1'
			sparseEncoding += tmp
		return sparseEncoding