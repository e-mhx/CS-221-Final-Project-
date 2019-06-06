import pickle
import re
import os
import numpy
from numpy import array
from sklearn.model_selection import train_test_split

import constants
import word_embeddings

with open("X.pkl", 'rb') as f:
    X = pickle.load(f) 
with open("y.pkl", 'rb') as f:
    y = pickle.load(f) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1000)

print(X_train)
print(y_train)

print(X_test)
print(y_test)


