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
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show
from sklearn import preprocessing

#Normalization function
def normalize(column):
    upper = column.max()
    lower = column.min()
    y = (column - lower)/(upper-lower)
    return y

# Load the Energy csv data using the Pandas library
filename = 'energydata_complete_nsm_appl.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 
#columns: from Lights until Tdewpoint
cols = range(2, 28) 
X = raw_data[:, cols]
appl = raw_data[:, 1]
appl = np.array(appl, dtype=np.float)

# Add class, application threshold <=60 and >60
y = []
for i in range (len(appl)):
    if (appl[i]<=60):
        y.append(0)
    else:
        y.append(1)
        
#thr1 = np.reshape(thr1, (-1, 1))
#X = np.concatenate((thr1,X),1)
y = np.array(y, dtype=np.float)
#attributeNames = np.insert(attributeNames, 0, [u'ApplClass'], axis=0)
#M = M+1   

N, M = X.shape
#Get the attributes names
attributeNames = np.asarray(df.columns[cols])

X = np.array(X, dtype=np.float)

##Standardize
#X = np.array(X, dtype=np.float)
#X = X - np.ones((N,1))*X.mean(axis=0)
##normalize each attribute by further dividing each attribute by its standard deviation
for c in range(M):
   stdv = X[:,c].std()
   for r in range(N):
       X[r,c] = X[r,c]/stdv

#normalize each attribute 
#for c in range(M):
#   X[:,c] = normalize(X[:,c])

