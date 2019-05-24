import pickle
import collections

COMMENT_INDEX = 17

with open('../data/condensed_dataset_SMALL.pkl', 'rb') as f:
    dataset = pickle.load(f) # dataset is the list of entries 

# creates a dictionry contain word counts for each entry
# these dictionaries will eventually be fed into a logistic classifier
for entry in dataset:
	entryDict = collections.defaultdict(int)

	entryList = entry[COMMENT_INDEX].split()
	for word in entryList:
		entryDict[word] += 1
