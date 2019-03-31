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
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from toolbox_02450 import rlr_validate

from basic import *

####################
#Linear regression
###################   
   
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = np.insert(attributeNames, 0, [u'Offset'], axis=0)
M = M+1   

# Values of lambda
#from 0.00001 to 100000000
lambdas = np.power(10.,range(-5,9))

internal_cross_validation = 10  

#Because this isn't two level, we will do a random split
# using stratification and 95 pct. split between training and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95)

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

print ("Optimal Lambda")
print (opt_lambda)
print ("Generalization error for optimal lambda")
print(opt_val_err)

title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

#MEAN OF train data
mu = np.mean(X_train[:, 1:], 0)
sigma = np.std(X_train[:, 1:], 0)

X_train[:, 1:] = (X_train[:, 1:] - mu ) / sigma
X_test[:, 1:] = (X_test[:, 1:] - mu ) / sigma

Xty = X_train.T @ y_train
XtX = X_train.T @ X_train

# Compute mean squared error without using the input data at all
Error_train_nofeatures = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
Error_test_nofeatures = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

# Estimate weights for the optimal value of lambda, on entire training set
lambdaI = opt_lambda * np.eye(M)
lambdaI[0,0] = 0 # Do no regularize the bias term
w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
# Compute mean squared error with regularization with optimal lambda
Error_train_rlr = np.square(y_train-X_train @ w_rlr).sum(axis=0)/y_train.shape[0]
Error_test_rlr = np.square(y_test-X_test @ w_rlr).sum(axis=0)/y_test.shape[0]

# Display results
print('Linear regression without feature selection:')
print('- R^2 train:     {0}'.format(Error_train_nofeatures))
print('- R^2 test:     {0}'.format(Error_test_nofeatures))
print('Regularized linear regression:')
print('- R^2 train:     {0}'.format(Error_train_rlr))
print('- R^2 test:     {0}'.format(Error_test_rlr))
   
   
   
   


