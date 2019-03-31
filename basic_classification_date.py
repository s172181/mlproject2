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
filename = 'energydata_cleaned.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 
#columns: from Lights until Tdewpoint
cols = range(1, 26) 
X = raw_data[:, cols]
appl = raw_data[:, 1]
appl = np.array(appl, dtype=np.float)

#Create the class array Y
#We will create classes based on the months, so 5 clases for the months
#january - may
classContent = raw_data[:,0]
dates = pd.DatetimeIndex(classContent)
classNames = np.unique(dates.month)
classDict = dict(zip(classNames,range(len(classNames))))
monthsnames = ["Jan","Feb","March","Apr","May"]
classDict2 = dict(zip(monthsnames,range(len(classNames))))

#Create the array y, this means that for each row, if the month is 1
# the class will be 0, if the month is 2, the class would be 1 etc
y = np.array([classDict[cl.month] for cl in dates])
#attributeNames = np.insert(attributeNames, 0, [u'ApplClass'], axis=0)
#M = M+1   

N, M = X.shape
#Get the attributes names
attributeNames = np.asarray(df.columns[cols])

