import pickle
import collections
import helper
import re 
import constants

top_20 = ['AskReddit', 'leagueoflegends', 'nba', 'funny', 'pics', 'nfl', 'pcmasterrace', \
    'videos', 'news', 'todayilearned', 'DestinyTheGame', 'worldnews', 'soccer', 'DotA2', \
    'AdviceAnimals', 'WTF', 'GlobalOffensive', 'hockey', 'movies', 'SquaredCircle']

with open(constants.SMALL_DATASET, 'rb') as f:
    dataset = pickle.load(f) 

trainSet, testSet = helper.partitionDataset(dataset)

wordProbByClass = helper.learnProbsMulti(trainSet)

countByClass, correctCountByClass = helper.classAndEvalMulti(testSet, wordProbByClass)

accuracyByClass = [float(correctCountByClass[i])/100 for i in range(constants.CATEGORY_COUNT)]

results = collections.defaultdict(int)
for i in range(len(top_20)):
	results[top_20[i]] = accuracyByClass[i]

print countByClass
print correctCountByClass
print results
