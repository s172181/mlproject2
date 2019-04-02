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
filename = 'energydata_complete_nsm.csv'
df = pd.read_csv(filename)
# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 

##
#Creating X and y
##
#Includes Tdewpoint
cols = range(2, 28) 
X = raw_data[:, cols]
X = np.array(X, dtype=np.float)
y = raw_data[:, 1]
y = np.array(y, dtype=np.float)
N, M = X.shape

datesContent = raw_data[:,0]
dates = pd.DatetimeIndex(datesContent)


#Get the attributes names
attributeNames = np.asarray(df.columns[cols])

####
#Normalization
####
#Standardize
X = X - np.ones((N,1))*X.mean(axis=0)
for c in range(M):
   stdv = X[:,c].std()
   for r in range(N):
       X[r,c] = X[r,c]/stdv
       
#normalize each attribute 
for c in range(M):
   X[:,c] = normalize(X[:,c])
