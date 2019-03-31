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

def sigmoid(x):
    e = np.exp(1)
    y = 1/(1+e**(-x))
    return y

# Load the Energy csv data using the Pandas library
filename = 'energydata_cleaned.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 

#columns: from Appliances until Tdewpoint
cols = range(1, 26) 
X = raw_data[:, cols]
X2 = raw_data[:, cols]
N, M = X.shape
#Get the attributes names
attributeNames = np.asarray(df.columns[cols])

std_scale = preprocessing.StandardScaler().fit(X)
df_std = std_scale.transform(X)

minmax_scale = preprocessing.MinMaxScaler().fit(X)
df_minmax = minmax_scale.transform(X)

print('Mean after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'
      .format(df_std[:,0].mean(), df_std[:,1].mean()))
print('\nStandard deviation after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'
      .format(df_std[:,0].std(), df_std[:,1].std()))

print (X)

####
#Normalization
####
#Lights has a skewed shape, so we will do a 

#Standardize
X2 = np.array(X2, dtype=np.float)
Xmeans = X2 - np.ones((N,1))*X2.mean(axis=0)
#normalize each attribute by further dividing each attribute by its standard deviation
for c in range(M):
   stdv = Xmeans[:,c].std()
   for r in range(N):
       Xmeans[r,c] = Xmeans[r,c]/stdv
       
#normalize each attribute 
for c in range(M):
   Xmeans[:,c] = normalize(Xmeans[:,c])


#df = pd.DataFrame(Xmeans)
#df[1] = np.log(df[1]+ 1)
#df[1] = normalize(df[1])
#print (df[1].describe() )
#n, bins, patches = plt.hist(df[1],bins = 15,alpha=1, histtype='bar', ec='black')