import pickle
import collections
import string
import helper
import re 

SMALL_DATASET = '../data/condensed_dataset_SMALL.pkl'
FULL_DATASET = '../../data/condensed_dataset.pkl'

CATEGORY_COUNT = 2
TESTSET_PROPORTION = 0.80

LAPLACE = 1
COMMENT_INDEX = 17
INENTRY_COUNT = 40000 # number of ingroup entries in the data set
OUTENTRY_COUNT = 760000

with open(FULL_DATASET, 'rb') as f:
    dataset = pickle.load(f) # dataset is the list of entries 

#splits dataset into test set and train set 


trainSet, testSet = helper.partitionDataset(dataset, CATEGORY_COUNT, TESTSET_PROPORTION)

ingroupWordProb, outgroupWordProb, ingroupEntryCount = helper.learnProbs(trainSet, LAPLACE, COMMENT_INDEX)

correctInCount, correctOutCount = helper.classAndEval(testSet, ingroupWordProb, outgroupWordProb, INENTRY_COUNT, OUTENTRY_COUNT, LAPLACE, COMMENT_INDEX)

# when run on FULL_DATASET
# 0/10,000
# 35/190,000
print correctInCount
print correctOutCount