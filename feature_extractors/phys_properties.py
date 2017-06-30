from misc import Misc

class PhysProperties:
	def getHeader(self):
		header = []
		for a in "A1 A2 A3 A4 A5 A6 A7 A8 A9".split(' '):
			for p in Misc.properties: 
				header.append("physProp_"+a+"-"+p)
		return header

	def getFeatures(self, l):
		features = []

		for a in l['aa']:
			features += Misc.physical_properties[a]

		return features