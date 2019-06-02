import pickle
import collections
import string
import re


tinyDataset = [
	["friend!", 0],
	["friend good!", 0], 
	["I love you so much", 0],
	["do you want to see a movie?", 0],
	["good friend", 0],
	["enemy!", 1],
	["enemy bad!", 1],
	["I hate you", 1],
	["get out of my sight", 1],
	["hate enemy", 1],
]

smallDataset = [
	["friend!", 0],
	["friend good!", 0], 
	["I love you so much", 0],
	["do you want to see a movie?", 0],
	["you are so important to me", 0],
	["there is no one I'd rather spend time with than you", 0],
	["we should see each other more", 0],
	["I'm so happy we got closer", 0],
	["happy important good", 0],
	["I want to spend time with you so much", 0],
	["enemy!", 1],
	["enemy bad!", 1],
	["I hate you", 1],
	["get out of my sight", 1],
	["evil exists because of people like you", 1],
	["I don't want to see you", 1],
	["I've never met anyone as incompetant as you", 1],
	["can you please leave me be", 1],
	["I hate you more than anyone", 1],
	["I will never like you", 1],
]

def partitionDataset(dataset, catCount, testProp): 
	trainSet = []
	testSet = []

	for i in range(len(dataset)):
		if i % (len(dataset)/catCount) >= (len(dataset)/catCount) * testProp:
			testSet.append(dataset[i])
		else:
			trainSet.append(dataset[i])

	return trainSet, testSet


def learnProbs(dataset, laplace, commentIdx):

	ingroupEntryCount = 0

	# get counts
	ingroupWordCount = collections.defaultdict(float)
	outgroupWordCount = collections.defaultdict(float)
	for entry in dataset:

		entryList = entry[commentIdx].split()
		if entry[-1] == 0: # if the current entry is classified in subreddit 1
			ingroupEntryCount += 1
			for word in entryList:
				normalizedWord = re.sub(r'[^\w\s]','',word.lower())
				ingroupWordCount[normalizedWord] += 1

		else: # if the current entry is NOT classified in subreddit 1
			entryList = entry[commentIdx].split()
			for word in entryList:
				normalizedWord = re.sub(r'[^\w\s]','',word.lower())
				outgroupWordCount[normalizedWord] += 1

	# calculate probabilities
	ingroupWordProb = collections.defaultdict(float)
	outgroupWordProb = collections.defaultdict(float)
	for key, val in ingroupWordCount.iteritems():
		ingroupWordProb[key] = (val+laplace)/(ingroupEntryCount + 2*laplace)

	for key, val in outgroupWordCount.iteritems():
		outgroupWordProb[key] = (val+laplace)/(len(dataset) - ingroupEntryCount + 2*laplace)

	return ingroupWordProb, outgroupWordProb, ingroupEntryCount

def classAndEval(testSet, ingroupWordProb, outgroupWordProb, ingroupEntryCount, outgroupEntryCount, laplace, commentIdx):
	
	correctInCount = 0
	correctOutCount = 0
	for entry in testSet:
		inWeight = 1.0
		outWeight = 1.0

		entryList = entry[commentIdx].split()
		for word in entryList:

			inWeight *= ingroupWordProb[word] if ingroupWordProb[word] != 0 else float(laplace)/(ingroupEntryCount + 2*laplace)
			outWeight *= outgroupWordProb[word] if outgroupWordProb[word] != 0 else float(laplace)/(outgroupEntryCount + 2*laplace)

		if inWeight > outWeight and entry[-1] == 0:
			correctInCount+=1
		if outWeight > inWeight and entry[-1] > 0:
			correctOutCount+=1

	return correctInCount, correctOutCount