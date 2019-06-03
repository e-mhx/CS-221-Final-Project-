import pickle
import collections
import helper
import re 
import constants

top_20 = ['AskReddit', 'leagueoflegends', 'nba', 'funny', 'pics', 'nfl', 'pcmasterrace', \
    'videos', 'news', 'todayilearned', 'DestinyTheGame', 'worldnews', 'soccer', 'DotA2', \
    'AdviceAnimals', 'WTF', 'GlobalOffensive', 'hockey', 'movies', 'SquaredCircle']

with open(constants.FULL_DATASET, 'rb') as f:
    dataset = pickle.load(f) 

trainSet, testSet = helper.partitionDataset(dataset)

wordProbByClass = helper.learnProbsMulti(trainSet)

countByClass, correctCountByClass = helper.classAndEvalMulti(testSet, wordProbByClass)

accuracyByClass = [float(correctCountByClass[i])/(constants.TRAINSET_LEN/constants.CATEGORY_COUNT) for i in range(constants.CATEGORY_COUNT)]

print (accuracyByClass[9] + accuracyByClass[10])/2

results = {value:key for value in top_20 for key in accuracyByClass}

print countByClass
print correctCountByClass
print results
