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

#columns: from Appliances until Tdewpoint
cols = range(1, 26) 
X = raw_data[:, cols]
N, M = X.shape
#Get the attributes names
attributeNames = np.asarray(df.columns[cols])

#Standardize
X = np.array(X, dtype=np.float)
Xmeans = X - np.ones((N,1))*X.mean(axis=0)
#normalize each attribute by further dividing each attribute by its standard deviation
for c in range(M):
   stdv = Xmeans[:,c].std()
   for r in range(N):
       Xmeans[r,c] = Xmeans[r,c]/stdv
       
#normalize each attribute 
for c in range(M):
   Xmeans[:,c] = normalize(Xmeans[:,c])

