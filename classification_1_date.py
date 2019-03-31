#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:10:01 2019

@author: alejuliet
"""

from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm

#Using appliance as class (class 1, appliance <=60, class2: appliance>60)
from basic_classification_date import *

#From january / february 7098
#X = X[:7098, :]
#y = y[:7098]

#From march / may 7099 - 15880
X = X[7099:15880, :]
y = y[7099:15880]
y2 = []
for i in range (len(y)):
    if (y[i]==2):
        y2.append(0)
    else:
        y2.append(1)
#####################
        
#thr1 = np.reshape(thr1, (-1, 1))
#X = np.concatenate((thr1,X),1)
y = np.array(y2, dtype=np.float)

# Fit logistic regression model
model = lm.logistic.LogisticRegression()
model = model.fit(X,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_white_prob = model.predict_proba(X)[:, 0] 

# Define a new data object (new type of wine), as in exercise 5.1.7
x = np.array([60,20,20.2,39.26,18.39,40,20.7,38.56,20.8933333333,37.6633333333,18.2,44.94,3.3333333333,81.6566666667,18.2,32.8266666667,20.29,42.3333333333,18.29,41.4633333333,3.7,744.1,93,5,2.6]).reshape(1,-1)
# Evaluate the probability of x being a white wine (class=0) 
x_class = model.predict_proba(x)[0,0]


# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

print('\nProbability of given sample being March: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_white_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_white_prob[class1_ids], '.r')
xlabel('Data object Month'); ylabel('Predicted prob. of January');
legend(['January', 'February'])
ylim(-0.01,1.5)

show()