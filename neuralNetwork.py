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

	# train the neural network
	def train(self, inputList, targetList):

		# calc output like query function
		inp = np.array(inputList, ndmin=2).T
		hInp = np.dot(self.Wih, inp)
		hOut = self.activationFunction(hInp)
		fInp = np.dot(self.Who, hOut)
		fOut = self.activationFunction(fInp)

		# calc final error
		target = np.array(targetList, ndmin=2).T
		fErr = target - fOut

		# calc hidden error
		hErr = np.dot(self.Who.T, fErr)

		# update ho layer
		self.Who += self.lr * np.dot((fErr * fOut * (1.0 - fOut)), np.transpose(hOut))

		# update ih layer
		self.Wih += self.lr * np.dot((hErr * hOut * (1.0 - hOut)), np.transpose(inp))

		pass

	pass
