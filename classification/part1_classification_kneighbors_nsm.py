#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:57:37 2019

@author: alejuliet
"""

from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from numpy import cov
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import model_selection

# Load Matlab data file and extract variables of interest
from basic_classification_nsm import *

classNames = ['midday','afternoon','night']


N, M = X.shape
C = len(classNames)

K = 5
CV = model_selection.KFold(K, shuffle=True)

# Maximum number of neighbors
L=20

test_error_rate_f = np.zeros(K)
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    
    print('Cross validation fold {0}/{1}'.format(k+1,K))
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    K2 = 5
    CV2 = model_selection.KFold(K2, shuffle=True)
    
    test_error_rate = np.zeros(K2)
    for (j, (trainj_index, testj_index)) in enumerate(CV2.split(X_train,y_train)):
        print('Cross iner fold {0}/{1}'.format(j+1,K2))
        Xj_train = X[trainj_index,:]
        yj_train = y[trainj_index]
        Xj_test = X[testj_index,:]
        yj_test = y[testj_index]
        
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        test_error_rate_0 = np.zeros(L)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(Xj_train, yj_train)
            y_est = knclassifier.predict(Xj_test)
            test_error_rate_0[l-1] = np.sum(y_est != yj_test) / len(yj_test)
            
        min_error = np.min(test_error_rate_0)
        opt_neighboor0 = np.argmin(test_error_rate_0)
        generror1 = (len(yj_test)/len(y_train))*min_error
        if (j == 0):
            generror = generror1
            knclassifier_op = knclassifier
            opt_neighboor = opt_neighboor0+1
        elif (generror1 < generror):
            generror = generror1
            knclassifier_op = knclassifier
            opt_neighboor = opt_neighboor0+1
            
    dist=2
    knclassifier_op.fit(X_train, y_train)
    y_est_f = knclassifier_op.predict(X_test)
    
    test_error_rate_f[k] = np.sum(y_est_f != y_test) / len(y_test)
    
    print("External cross")
    
    print("Fold",k)
    print("Neighboor ",opt_neighboor)
    print("Test error rate ",test_error_rate_f[k])
        
    
    
    
    
    
    
    