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
filename = 'energydata_cleaned_orderedappliance.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 
#columns: from Lights until Tdewpoint
cols = range(2, 26) 
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

####
#Normalization
####
#Standardize

