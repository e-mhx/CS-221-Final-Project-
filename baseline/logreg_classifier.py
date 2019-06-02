import pickle
import collections
import string
import helper
import re 

with open('../data/condensed_dataset_SMALL.pkl', 'rb') as f:
    dataset = pickle.load(f) # dataset is the list of entries 

ingroupWordProb, outgroupWordProb = helper.learnProbs(dataset)

# TODO: when infering, use laplace !!