import pandas as pd
from sklearn.model_selection import train_test_split

class Splitter:
	def __init__(self, data, penguin):
		self.data = data
		# If splitting penguin datset, penguin = True
		self.penguin = penguin

	def split(self):
		if self.penguin:
			# Features column, remove species as its not a feature
			features = self.data.drop(columns = ['species'])
			# Target column
			target = self.data['species']
		else:
			features = self.data.drop(columns = ['Type'])
			target = self.data['Type']

		# Split the dataset into a training set and a test set with default parameter values
		features_train, features_test, target_train, target_test= train_test_split(features, target)

		return features_train, features_test, target_train, target_test