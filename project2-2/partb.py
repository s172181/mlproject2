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
import torch
import pandas as pd
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from numpy import array

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
cols = range(2, 28) 
X = raw_data[:, cols]
X = np.array(X, dtype=np.float)
#y = raw_data[:, 1]
y = raw_data[:,[1]] 
y = np.array(y, dtype=np.float)
N, M = X.shape
attributeNames = np.asarray(df.columns[cols])
attributeNames = list( attributeNames )

#Standardize
X = X - np.ones((N,1))*X.mean(axis=0)
for c in range(M):
   stdv = X[:,c].std()
   for r in range(N):
       X[r,c] = X[r,c]/stdv
       
#normalize each attribute 
#for c in range(M):
#   X[:,c] = normalize(X[:,c])
       
# Regression
#lambda
#from 0.00001 to 100000000
lambdas = np.power(10.,range(-10,10))
   
#Neural network
# Parameters for neural network classifier
n_hidden_units = 2      # number of hidden units
hiddenUnits = [10,15,20,26]
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000        # 

# Define the model
#model = lambda: torch.nn.Sequential(
#                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
#                    torch.nn.Tanh(),   # 1st transfer function,
#                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
#                    # no final tranfer function, i.e. "linear output"
#                    )
#loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

# K-fold crossvalidation
K = 10                 
CV = model_selection.KFold(K, shuffle=True)


errorsf = [] 
h = 0
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train = torch.tensor(y[train_index], dtype=torch.float)
    X_test = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test = torch.tensor(y[test_index], dtype=torch.float)
    internal_cross_validation = 10 
    
    #Internal loop
    K2 = 10                 
    CV2 = model_selection.KFold(K2, shuffle=True)
    
    model2 = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, hiddenUnits[h]), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(hiddenUnits[h], 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
    loss_fn2 = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    print('Training model of type:\n\n{}\n'.format(str(model2())))
    
    errors = [] # make a list for storing generalizaition error in each loop
    for (j, (trainj_index, testj_index)) in enumerate(CV2.split(X_train,y_train)): 
        print("\n\tInternal cross")
        # Extract training and test set for current CV fold, convert to tensors
        Xj_train = torch.tensor(X[trainj_index,:], dtype=torch.float)
        yj_train = torch.tensor(y[trainj_index], dtype=torch.float)
        Xj_test = torch.tensor(X[testj_index,:], dtype=torch.float)
        yj_test = torch.tensor(y[testj_index], dtype=torch.float)
        
    
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model2,
                                                           loss_fn2,
                                                           X=Xj_train,
                                                           y=yj_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
    
        
        print('\n\tBest loss: {}\n'.format(final_loss))
        # Determine estimated class labels for test set
        yj_test_est = net(Xj_test)
        # Determine errors and errors
        se = (yj_test_est.float()-yj_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(yj_test)).data.numpy() #mean
        errors.append(mse) # store error rate for current CV fold 
        generror1 = (len(yj_test)/len(y_train))*mse
        if (j == 0):
            generror = generror1
        elif (generror1 < generror):
            bestmodel = lambda: net
            generror = generror1
    
    
    
    # Train the net on training data
    netf, final_lossf, learning_curvef = train_neural_net(bestmodel,
                                                       loss_fn2,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    # Determine estimated class labels for test set
    y_test_est = netf(X_test)
    
    # Determine errors and errors
    se2 = (y_test_est.float()-y_test.float())**2 # squared error
    mse2 = (sum(se2).type(torch.float)/len(y_test)).data.numpy() #mean
    errorsf.append(mse2) # store error rate for current CV fold 
    print("External cross")
    
    print("Fold",k)
    print("Hidden unit",hiddenUnits[h])
    print("Test error RMSE",np.sqrt(mse2))
    print("Best error from internal cross",generror)
    
    h+= 1
    if (h==4):
        h = 0

#print(errorsf)
    
    


