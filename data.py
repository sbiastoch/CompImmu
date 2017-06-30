'''
	Container-Class which holds the data and accumulates all features.
'''

from misc import Misc

class Data:
	csv = []
	featureExtractors = []
	output_buffer = ""

	def loadFromFile(self, filepath):
		f = open(filepath, 'r')
		f.readline() # Skip first line with headers
		for line in f:
			fields = line.strip().split("\t")
			aa = list(fields[0])
			c50 = fields[1]
			binder = fields[2]
			self.csv.append({"aa": aa, "c50": c50, "class": binder})
		return self

	def loadFromArray(self, peptides):
		self.csv = [{'aa': aa} for aa in peptides]
		return self

	# Adds a class or an array of classes of feature-extractors, e.g. AminoAcids or IC50 or SparseEncoding
	def setFeatures(self, feature_classes):
		self.featureExtractors = feature_classes if isinstance(feature_classes, list) else [feature_classes]
		return self

	# Format data in CSV-style. No data is printed or returned, use puts() or save() instead.
	def formatCSV(self):
		ret = [",".join(self._getHeader())]
		for l in self.csv:
			ret.append(",".join(self._getDataRow(l)))

		self.output_buffer = "\n".join(ret)
		return self

	def formatTensorflowCSV(self):

		rows = len(self.csv)
		feature_dims = len(self._getHeader())-1
		ret = [str(rows) + "," + str(feature_dims) + "," + "binder,non-binder"]
		for l in self.csv:
			ret.append(",".join(self._getDataRow(l)))

		self.output_buffer = "\n".join(ret)+"\n"

		return self

	# Prints the current output buffer.
	def puts(self):
		if(not self.output_buffer):
			print "Unformatted or empty output buffer! Format first using e.g. formatCSV()."
		print self.output_buffer
		return self

	# Saves the current output buffer to file, file is created if not exists, otherwise overwritten
	def save(self, filepath):
		if(not self.output_buffer):
			print "Unformatted or empty output buffer! Format first using e.g. formatCSV()."
			return self

		f = open(filepath, 'w+')
		f.write(self.output_buffer)
		f.closed
		return self

	# Formats the output buffer
	def toArray(self):
		ret = [self._getHeader()]
		for l in self.csv:
			ret.append(self._getDataRow(l))

		return ret

	# Only feature extractors returning numeric values are allowed here
	def toFeatureArray(self):
		rows = []
		for l in self.csv:
			for f in self.featureExtractors:
				rows.append([float(x) for x in f.getFeatures(l)])

		return rows




	###       ###       ###
	### PRIVATE METHODS ###
	###       ###       ###




	# Accumulates all headers of feature extractors in the order of adding
	# Appends an 'class' column at the very end
	# Returns an array of strings
	def _getHeader(self):
		header = []
		for f in self.featureExtractors:
			header += f.getHeader()
		header.append('class')
		return header

	# Accumulates all features of given feature-extractors in the order of 
	# adding for a single data-line (hashmap)
	# Returns an array of strings
	def _getDataRow(self, l):
		row = []
		for f in self.featureExtractors:
			row += f.getFeatures(l)
		row += [l['class']]

		return row