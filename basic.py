#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:20:15 2019

@author: 
    Alechandrina Pereira s172181
    Maciej Maj, s171706
    Sushruth Bangre, s190021
"""

import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#Normalization function
def normalize(column):
    upper = column.max()
    lower = column.min()
    y = (column - lower)/(upper-lower)
    return y

# Load the Energy csv data using the Pandas library
filename = 'energydata_cleaned.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 

#columns: from Lights until Tdewpoint
cols = range(2, 26) 
X = raw_data[:, cols]
y = raw_data[:, 1]
y = np.array(y, dtype=np.float)
N, M = X.shape
#Get the attributes names
attributeNames = np.asarray(df.columns[cols])

####
#Normalization
####
#Standardize
X = np.array(X, dtype=np.float)
X = X - np.ones((N,1))*X.mean(axis=0)
#normalize each attribute by further dividing each attribute by its standard deviation
for c in range(M):
   stdv = X[:,c].std()
   for r in range(N):
       X[r,c] = X[r,c]/stdv
       
#normalize each attribute 
for c in range(M):
   X[:,c] = normalize(X[:,c])

