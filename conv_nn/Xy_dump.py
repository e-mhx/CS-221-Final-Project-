import pickle
import re
import os
from numpy import array

import constants
import word_embeddings

# # X is a list (numpy.ndarray) of 150 entries, where each entry is a R^4 feature vector
# # y is a list (numpy.ndarray) of 150 classification labels, where y[i] is the label for X[i]
# X, y = load_iris(return_X_y=True)

# print(X)
# print(y)


with open(constants.SMALL_DATASET, 'rb') as f:
    dataset = pickle.load(f) 
comments = [entry[constants.COMMENT_INDEX].split() for entry in dataset]
#---------------------------------------------

wordToVec = word_embeddings.getWordEmbeddingDict(10000)

X = []
y = []
for k in range(len(comments)):
	comment = comments[k]

	commentVecAdditions = 0
	commentVec = [0.0 for _ in range(constants.WORDVEC_LEN)]

	for i in range(len(comment)): # iterates through each word of the comment
		while i < len(comment) and wordToVec[re.sub(r'[^\w\s]','',comment[i].lower())] == 0: 
			i+=1
		if i == len(comment): continue

		wordVec = wordToVec[re.sub(r'[^\w\s]','',comment[i].lower())]
		#print(wordVec[3])
		commentVec = [commentVec[i] + wordVec[i] for i in range(constants.WORDVEC_LEN)]
		commentVecAdditions += 1

	if i == len(comment): continue
	commentVec = [commentVec[i]/commentVecAdditions for i in range(len(commentVec))]

	X.append(commentVec)
	y.append(dataset[k][-1])
	

X = array(X)
y = array(y)


with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('y.pkl', 'wb') as f:
    pickle.dump(y, f)