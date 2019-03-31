#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:10:01 2019

@author: alejuliet
"""

from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm

#Using appliance as class (class 1, appliance <=60, class2: appliance>60)
from basic_classification import *

# Fit logistic regression model
model = lm.logistic.LogisticRegression()
model = model.fit(X,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_white_prob = model.predict_proba(X)[:, 0] 

# Define a new data object (new type of wine), as in exercise 5.1.7
x = np.array([0,17.29,42.7,16.39,42.9,17.7,40.9,15.39,42.09,15.6,48.79,5.59,99.9,15.7694444444,37.59,16.705,45.4055555556,15.19,41.9,5.0666666667,766.1666666667,99.6666666667,3.6666666667,5.0333333333]).reshape(1,-1)
# Evaluate the probability of x being a white wine (class=0) 
x_class = model.predict_proba(x)[0,0]


# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

print('\nProbability of given sample being <60: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_white_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_white_prob[class1_ids], '.r')
xlabel('Data object Appliance'); ylabel('Predicted prob. of class < 60');
legend(['<=60', '>60'])
ylim(-0.01,1.5)

show()