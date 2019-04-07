#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: 
    Alechandrina Pereira s172181
    Maciej Maj, s171706
    Sushruth Bangre, s190021
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from sklearn import model_selection

font_size = 15
plt.rcParams.update({'font.size': font_size})

from basic_classification_nsm import *

classNames = ['midday','afternoon','night']

C = len(classNames)

K = 5

CV = model_selection.KFold(K, shuffle=True)

train_error_rate_f = np.zeros(K)
test_error_rate_f = np.zeros(K)
coefficient_norm_f = np.zeros(K)
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    
    print('Cross validation fold {0}/{1}'.format(k+1,K))
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    K2 = 5
    CV2 = model_selection.KFold(K2, shuffle=True)
    
    #lambda_interval = np.logspace(-8, 1, 10)
    lambda_interval = np.logspace(-1, 2, 10)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    super_lambda = 0
    for (j, (trainj_index, testj_index)) in enumerate(CV2.split(X_train,y_train)):
        Xj_train = X[trainj_index,:]
        yj_train = y[trainj_index]
        Xj_test = X[testj_index,:]
        yj_test = y[testj_index]
    
        for h in range(0, len(lambda_interval)):
            mdl = LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                               tol=1e-4, random_state=1, 
                               penalty='l2', C=1/lambda_interval[h],max_iter=700 )
            
            mdl.fit(Xj_train, yj_train)
        
            y_train_est = mdl.predict(Xj_train).T
            y_test_est = mdl.predict(Xj_test).T
            
            train_error_rate[h] = np.sum(y_train_est != yj_train) / len(yj_train)
            test_error_rate[h] = np.sum(y_test_est != yj_test) / len(yj_test)
            
        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        
        generror1 = (len(yj_test)/len(y_train))*min_error
        if (j == 0):
            generror = generror1
            super_lambda = opt_lambda
        elif (generror1 < generror):
            super_lambda = opt_lambda
            generror = generror1
            
    mdl2 = LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                               tol=1e-4, random_state=1, 
                               penalty='l2', C=1/super_lambda,max_iter=700 )
    mdl.fit(X_train, y_train)
    
    y_test_est_f = mdl.predict(X_test).T
    
    test_error_rate_f[k] = np.sum(y_test_est_f != y_test) / len(y_test)
    
    print("External cross")
    
    print("Fold",k)
    print("Lambda",super_lambda)
    print("Test error rate ",test_error_rate_f[k])
    
    
            
    
        
        
        
        
