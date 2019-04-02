Things to mention:

1. Tdewpoint is included because it seems it is important (it is on the report)

Index of files:

1. addNSMcolumn.ipynb : it is a jupyter notebook file (it is python but using notebook, I just used it to create the extra column NSM with the seconds)
2. basic.py: loads the data, creates X and y and standardize and normalize the data
3. energydata_complete_nsm.csv: this is the file that contains the extra NSM column
4. linearregression_basic.py: it does the most simple linear regression and shows the graph of Appliance vs residual (Maybe we can include this in the report)
6. linearregression_crossvalidation.py: applies 10-fold cross-validation to the problem of fitting a linear regression model and outputs the mean squared error
7. For Part a, question 2: linearregression_crossvalidation_lambda.py: this include the lambdas (it follows exactly the exercise 8_1_1, I just added comments and the RMSE, as well as printing the optimal lambda for every iteration). Feel free to tune in the lambdas, I honestly didn't find much different between regularizing or not, because it choose as optimal lambda 1.
8. For Part a, question 2: linearregression_featureselector.py: this does the feature selection (it takes a while), but honestly the test error doesnt seem to improve. However, it does seem that the NSM attribute is always chosen.

Things to notice:

1. Some of the scripts use "toolbox_02450". Please, add it to the folder when you put the scripts. 
2. np.square(y_true-y_est).sum()/len(y_true) is the mean squared error
3. In linearegresion scripts: RMSE is the Root Mean Square Error. In the report "Data driven prediction models of energy use of appliances in a
low-energy house" they use this instead, and looking at the results, is the same as ours in the case of "linearregression_basic"
