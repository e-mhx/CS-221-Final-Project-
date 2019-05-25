# import pytorch-neural-network as nn
import importlib
import sys
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
np.version.version
from random import shuffle
pnn = importlib.import_module("pytorch-neural-network")
we = importlib.import_module("word_embeddings")

# Global paramters
wordEmbedDict = we.getWordEmbeddingDict() # load the dictionary
keyset = wordEmbedDict.keys()
# TRAINING_EXAMPLES = enter some number here
# index of true label
TRUE_LABEL = 8 #index of the true label, hard coded
NUM_FEATURES = 300
NUM_EXAMPLES = 3
BODY_INDEX = 17 # hard coded because of the structure of our depickled data
unparsed = "../data/condensed_dataset_SMALL.pkl" # will change this so we can just call the pkl set

def loadDataSet(unparsedDataSet):
	with open(unparsedDataSet, 'rb') as f:
		dataset = pickle.load(f)
	return dataset

def vectorizeWord(word):
	vWord = np.zeros((NUM_FEATURES, 1))
	if word in keyset:
		vWord = np.array([wordEmbedDict[word]]).reshape((NUM_FEATURES, 1))
	return vWord

def vectorizeComment(body):
	# initialize empty sum
	vComment = np.zeros((NUM_FEATURES, 1))
	words = body.split()
	for word in words:
		vWord = vectorizeWord(word)
		vComment = np.add(vComment, vWord)
	return vComment

# Vectorize pickled data into usable format, return X with m examples with NUM_FEATURES features
def vectorizeDataSet(data, m):
	unrollComment = data[0][BODY_INDEX]
	X = vectorizeComment(unrollComment)

	for i in range(1, m):
		comment = data[i][BODY_INDEX]
		example = vectorizeComment(comment)
		X = np.append(X, example, axis = 1)

	print("Size of X: ", X.shape)
	return X

# def getYValueo

def main():
	# Goal: single hidden layer neural network with 3 neurons in it

	# TODO: Paramterize neural network
	# m = TRAINING_EXAMPLES
	# numNeurons
	# numHiddenLayers

	# Loads pickle, parses it, and returns our X
	# Each column is a feature vector for an entire comment, with NUM_EXAMPLES columns
	# Each row represents some feature of the comment (unknown)
	dataSet = loadDataSet(unparsed) 
	shuffle(dataSet)
	X = vectorizeDataSet(dataSet, 3)
	# dataSet[1:3][1:2]
	# npX = transformData(X)
	# vector = vectorizeComment(comment)
	# print(vector)
	
	# 
	# word = 'hi'
	# for i in range(4):
	# 	thingToPrint = dataSet[i][TRUE_LABEL]
	# 	print("---------------------------\n\n\n\n-----------------------------")
	# 	print(thingToPrint)
	# len(wordEmbedDict[word])

	
	
	# print("size of feature vector for the word ", word, ": ", thingToPrint)
	NN = pnn.Neural_Network()

	# In this iteration X is our training set and xPredicted is our test set
	# Will implement validation set soon
	# TODO: Validation set

	# TODO: Change these to a data set X where each column is a feature with m training examples
	# Dummy data for now to implement 
	# Create data set and labels
	X = torch.tensor(([4, 9], [1, 5], [3, 6]), dtype=torch.float) # 3 X 2 tensor # 3 X 2 tensor, this is our entire data set
																	# but will be changed to training set
	y = torch.tensor(([72], [100], [85]), dtype=torch.float) # 3 X 1 tensor
	xPredicted = torch.tensor(([4, 8]), dtype=torch.float) # 1 X 2 tensor, this is the test set

	# scale units
	# TODO: Scale by subtracting the mean and divide by the standard deviation
	X_max, _ = torch.max(X, 0)
	xPredicted_max, _ = torch.max(xPredicted, 0)
	X = torch.div(X, X_max)
	xPredicted = torch.div(xPredicted, xPredicted_max)
	y = y / 100  # max test score is 100	
	for i in range(1000):  # trains the NN 1,000 times
		# This print statement outputs the value of our loss function. If it is decreasing, then we are doing well
	    # print("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
	    NN.train(X, y)


	NN.saveWeights(NN)
	NN.predict(xPredicted)

main()
