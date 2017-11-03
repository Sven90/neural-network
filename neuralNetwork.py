import numpy as np
import scipy.special

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
		self.Who = np.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))
		
		# define activation function
		self.activationFunction = lambda x: scipy.special.expit(x)
		
		pass

	# train
	def train():

		pass

	# query the neural network query([1, 3, 4, 5, ...])
	def query(self, inputList):

		# convert input list to vector
		inp = np.array(inputList, ndmin=2).T

		# calc first cycle
		hInp = np.dot(self.Wih, inp)
		hOut = self.activationFunction(hInp)

		# calc final cycle
		fInp = np.dot(self.Who, hOut)
		fOut = self.activationFunction(fInp)

		return fOut

		pass

	pass

# tests

n = neuralNetwork(3, 3, 3, 0.3)
print(n.query([1.0, 0.5, -1.5]))
