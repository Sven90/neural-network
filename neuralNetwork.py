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
		# define inverse activation function
		self.inverseActivationFunction = lambda x: spec.logit(x)
		
		pass

	# query the neural network, query([1, 3, 4, 5, ...])
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

	# backquery the neural network
	def backquery(self, targetList):

		targets = np.array(targetList, ndmin=2).T
		finalOutputVector = self.inverseActivationFunction(targets)
		hiddenOutputVector = np.dot(self.WeightsHiddenOutput.T, finalOutputVector)
		# scale back
		hiddenOutputVector -= np.min(hiddenOutputVector)
		hiddenOutputVector /= np.max(hiddenOutputVector)
		hiddenOutputVector *= 0.98
		hiddenOutputVector += 0.01

		hiddenInputVector = self.inverseActivationFunction(hiddenOutputVector)
		inputVector = np.dot(self.WeightsInputHidden.T, hiddenInputVector)
		inputVector -= np.min(inputVector)
		inputVector /= np.max(inputVector)
		inputVector *= 0.98
		inputVector += 0.01

		return inputVector

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

# train neural network for handwritten number recognition

inputNodes = 784
hiddenNodes = 100
outputNodes = 10
learningRate = 0.2

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

trainingDataFile = open("mnist_train.csv", 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close()

for record in trainingDataList:
	allValues = record.split(',')
	inputs = (np.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
	targets = np.zeros(outputNodes) + 0.01
	targets[int(allValues[0])] = 0.99
	n.train(inputs, targets)

	pass

# test it
scorecard = []
testDataFile = open("mnist_test.csv", 'r')
testDataList = testDataFile.readlines()
testDataFile.close()

epochs = 1

for e in range(epochs):
	for record in testDataList:
		allValues = record.split(',')
		correctAnswer = int(allValues[0])
		#print(correctAnswer, "correct answer")
		inputs = (np.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
		outputs = n.query(inputs)
		answer = np.argmax(outputs)
		#print(answer, "network's answer")
		if (answer == correctAnswer):
			scorecard.append(1)
		else:
			scorecard.append(0)
			pass

		pass

	pass	

# calculate performance
scorecardArray = np.asfarray(scorecard)
print("performance = ", scorecardArray.sum() / scorecardArray.size)

# look at the neural networks brain
answer = 2
targets = np.zeros(outputNodes) + 0.01
targets[answer] = 0.99
print(targets)
imageData = n.backquery(targets)

#plot brain
plt.imshow(imageData.reshape(28,28), cmap='Greys', interpolation='None')
plt.show()
