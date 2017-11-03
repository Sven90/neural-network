import numpy as np

# neural network class
class neuralNetwork:

	# initialise neural network with its parameters
	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):

		# initialise sizes
		self.iNodes = inputNodes
		self.hNodes = hiddenNodes
		self.oNodes = outputNodes
		self.lr = learningRate

		# make weight matrices
		self.Wih = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
		self.Who = np.random.normal(0.0, pow(self,oNodes, -0.5), (self.oNodes, self.hNodes))
		pass

	# train
	def train():

		pass

	#query
	def query():

		pass

	pass
