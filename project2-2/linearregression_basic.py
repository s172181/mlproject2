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
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from toolbox_02450 import rlr_validate
import sklearn.linear_model as lm
from sklearn import metrics

from basic import *

####################
#Linear regression
###################   
   
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = np.insert(attributeNames, 0, [u'Offset'], axis=0)
M = M+1   

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict appliance / energy compsumtion
y_est = model.predict(X)
residual = y-y_est

# Display scatter plot
figure()
figure(0)
subplot(2,1,1)
plot(y, residual, '.')
xlabel('Appliance (true)'); ylabel('Appliance (estimated)');
figure(1)
subplot(2,1,2)
hist(residual,40)
xlabel('Residual');

#Mean squared error
print (np.sqrt(np.square(y-y_est).sum()/len(y)))

print(metrics.mean_squared_error(y,y_est))
#Which is the same as
print( np.square(y-y_est).sum()/len(y))

print ("RMSE")
print (np.sqrt(np.square(y-y_est).sum()/len(y)))

show()
   
   
   


