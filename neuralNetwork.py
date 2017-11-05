# imports
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt

# neural network class
class neuralNetwork:

	# initialise neural network with its parameters
	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):

		# initialise sizes
		self.inputNodes = inputNodes
		self.hiddenNodes = hiddenNodes
		self.outputNodes = outputNodes
		self.learningRate = learningRate

		# make weight matrices
		self.WeightsInputHidden = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
		self.WeightsHiddenOutput = np.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))
		
		# define activation function
		self.activationFunction = lambda x: spec.expit(x)
		
		pass

	# query the neural network like query([1, 3, 4, 5, ...])
	def query(self, inputList):

		# convert input list to vector
		inputVector = np.array(inputList, ndmin=2).T

		# calc first cycle
		hiddenInputVector = np.dot(self.WeightsInputHidden, inputVector)
		hiddenOutputVector = self.activationFunction(hiddenInputVector)

		# calc final cycle
		finalInputVector = np.dot(self.WeightsHiddenOutput, hiddenOutputVector)
		finalOutputVector = self.activationFunction(finalInputVector)

		return finalOutputVector

	# train the neural network
	def train(self, inputList, targetList):

		# calc output like query function
		inputVector = np.array(inputList, ndmin=2).T
		hiddenInputVector = np.dot(self.WeightsInputHidden, inputVector)
		hiddenOutputVector = self.activationFunction(hiddenInputVector)
		finalInputVector = np.dot(self.WeightsHiddenOutput, hiddenOutputVector)
		finalOutputVector = self.activationFunction(finalInputVector)

		# calc final error
		target = np.array(targetList, ndmin=2).T
		finalError = target - finalOutputVector

		# calc hidden error
		hiddenError = np.dot(self.WeightsHiddenOutput.T, finalError)

		# update final layer
		self.WeightsHiddenOutput += self.learningRate * np.dot((finalError * finalOutputVector * (1.0 - finalOutputVector)), np.transpose(hiddenOutputVector))

		# update first layer
		self.WeightsInputHidden += self.learningRate * np.dot((hiddenError * hiddenOutputVector * (1.0 - hiddenOutputVector)), np.transpose(inputVector))

		pass

	pass

# train neural network for handwritten number recognition


