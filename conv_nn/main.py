import pickle
import re
import os
import numpy
from numpy import array
from sklearn.linear_model import LogisticRegression

import constants
import word_embeddings

with open("X.pkl", 'rb') as f:
    X = pickle.load(f) 
with open("y.pkl", 'rb') as f:
    y = pickle.load(f) 

print(X)
print(y)