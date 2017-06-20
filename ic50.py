class IC50:
	def getHeader(self):
		return ["C50"]

	def getFeatures(self, data_vector):
		return [data_vector['c50']]
