from misc import Misc

class Data:
	csv = []
	featureExtractors = []
	output = ""

	def __init__(self, filepath):
		f = open(filepath, 'r')
		f.readline() # Skip first line with headers
		for line in f:
			fields = line.strip().split("\t")
			aa = list(fields[0])
			c50 = fields[1]
			binder = fields[2]
			self.csv.append({"aa": aa, "c50": c50, "class": binder})


	def addFeatures(self, feature_classes):
		self.featureExtractors += feature_classes if isinstance(feature_classes, list) else [feature_classes]
		return self


	def formatCSV(self):
		ret = ["\t".join(self._getHeader())]
		for l in self.csv:
			ret.append("\t".join(self._getDataRow(l)))

		self.output = "\n".join(ret)
		return self


	def puts(self):
		print self.output
		return self


	def save(self, filepath):
		f = open(filepath, 'w+')
		f.write(self.output)
		f.closed
		return self


	def _getHeader(self):
		header = []
		for f in self.featureExtractors:
			header += f().getHeader()
		header.append('class')
		return header


	def _getDataRow(self, l):
		row = []
		for f in self.featureExtractors:
			row += f().getFeatures(l)
		row += [l['class']]

		return row