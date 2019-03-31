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

from basic import *

####################
#Linear regression
###################   
   
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = np.insert(attributeNames, 0, [u'Offset'], axis=0)
M = M+1   

# Additional nonlinear attributes, t_out and rh_out squared
t_out_idx = 21
rh_out_idx = 23
Xfa2 = np.power(X[:,t_out_idx],2).reshape(-1,1)
Xva2 = np.power(X[:,rh_out_idx],2).reshape(-1,1)
Xfava = (X[:,t_out_idx]*X[:,rh_out_idx]).reshape(-1,1)
X = np.asarray(np.bmat('X, Xfa2, Xva2, Xfava'))

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict appliance / energy compsumtion
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
figure(0)
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Appliance (true)'); ylabel('Appliance (estimated)');
figure(1)
subplot(2,1,2)
hist(residual,40)
xlabel('Residual');

show()
   
   
   


