import pickle
import collections
import string
import re

COMMENT_INDEX = 17
LAPALCE = 1

testDataset = [
	["friend!", 0],
	["friend good!", 0], 
	["enemy!", 1],
	["enemy bad!", 1]
]

def learnProbs(dataset):

	ingroupEntryCount = 0

	# get counts
	ingroupWordCount = collections.defaultdict(float)
	outgroupWordCount = collections.defaultdict(float)
	for entry in dataset:

		entryList = entry[COMMENT_INDEX].split()
		if entry[-1] == 0: # if the current entry is classified in subreddit 1
			ingroupEntryCount += 1
			for word in entryList:
				normalizedWord = re.sub(r'[^\w\s]','',word.lower())
				ingroupWordCount[normalizedWord] += 1

		else: # if the current entry is NOT classified in subreddit 1
			entryList = entry[COMMENT_INDEX].split()
			for word in entryList:
				normalizedWord = re.sub(r'[^\w\s]','',word.lower())
				outgroupWordCount[normalizedWord] += 1

	# calculate probabilities
	ingroupWordProb = collections.defaultdict(float)
	outgroupWordProb = collections.defaultdict(float)
	for key, val in ingroupWordCount.iteritems():
		ingroupWordProb[key] = (val+LAPALCE)/(ingroupEntryCount + 2*LAPALCE)

	for key, val in outgroupWordCount.iteritems():
		outgroupWordProb[key] = (val+LAPALCE)/(len(dataset) - ingroupEntryCount + 2*LAPALCE)

	return ingroupWordProb, outgroupWordProb