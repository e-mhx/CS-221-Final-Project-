import pickle
import collections
import string
import re
import constants

def partitionDataset(dataset): 
	trainSet = []
	testSet = []

	for i in range(len(dataset)):
		if i % (len(dataset)/constants.CATEGORY_COUNT) >= (len(dataset)/constants.CATEGORY_COUNT) * constants.TESTSET_PROPORTION:
			testSet.append(dataset[i])
		else:
			trainSet.append(dataset[i])

	return trainSet, testSet


def learnProbsMulti(trainSet):

	wordCountByClass = []
	for _ in range(constants.CATEGORY_COUNT): wordCountByClass.append(collections.defaultdict(float))

	for entry in trainSet:
		entryList = entry[constants.COMMENT_INDEX].split()
		for word in entryList:
			normalizedWord = re.sub(r'[^\w\s]','',word.lower())
			wordCountByClass[entry[-1]][normalizedWord] += 1

	wordProbByClass = []
	for _ in range(constants.CATEGORY_COUNT): wordProbByClass.append(collections.defaultdict(float))

	for i in range(constants.CATEGORY_COUNT):
		for key, val in wordCountByClass[i].iteritems():
			wordProbByClass[i][key] = (val+constants.LAPLACE)/(len(trainSet)/constants.CATEGORY_COUNT + 2*constants.LAPLACE)

	# ignore the most common words with little semantic meaning 
	topWords = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", \
		"this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "will", "an", "my", "one", "all", "would", "there", "their", "what"]
	for i in range(constants.CATEGORY_COUNT):
		for word in topWords:
			wordProbByClass[i][word] = 1.0

	return wordProbByClass


def classAndEvalMulti(testSet, wordProbByClass):
	countByClass = [0 for _ in range(constants.CATEGORY_COUNT)]
	correctCountByClass = [0 for _ in range(constants.CATEGORY_COUNT)]

	for i in range(len(testSet)):
		entry = testSet[i]
		weightByClass = [1.0 for _ in range(constants.CATEGORY_COUNT)]

		entryList = entry[constants.COMMENT_INDEX].split()
		for word in entryList:
			for i in range(constants.CATEGORY_COUNT):
				weightByClass[i] *= wordProbByClass[i][word] if wordProbByClass[i][word] != 0 else float(constants.LAPLACE)/(constants.TRAINSET_LEN/constants.CATEGORY_COUNT + 2*constants.LAPLACE)

		classification = weightByClass.index(max(weightByClass))

		countByClass[classification] += 1

		if entry[-1] == classification:
			correctCountByClass[classification] += 1

	return countByClass, correctCountByClass
