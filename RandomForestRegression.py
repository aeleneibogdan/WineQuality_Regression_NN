# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:09:24 2021

@author: aelen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree

import os
import pydotplus
import six

datasetwhite=pd.read_csv('winequality-white.csv',sep=';',header=0)

datasetred=pd.read_csv('winequality-red.csv',sep=';',header=0)


#splitting the last column from the entire white wine dataset
xWhite=datasetwhite.iloc[:,:11]
tWhite=datasetwhite['quality']

#splitting the last column from the entire red wine dataset
xRed=datasetred.iloc[:,:11]
tRed=datasetred['quality']

# sb.pairplot(datasetwhite, hue='quality',height=1.5,palette="PuBu")
# sb.pairplot(datasetred, hue='quality',height=1.5)

#splitting the white wine dataset
xTrainWhite, xTestWhite, tTrainWhite, tTestWhite = train_test_split(xWhite, tWhite, test_size=0.2)

#scaling the white wine dataset
scaler = StandardScaler()
scaler.fit(xTrainWhite)
xTrainWhite = scaler.transform(xTrainWhite)
xTestWhite = scaler.transform(xTestWhite)

#splitting the red wine dataset
xTrainRed, xTestRed, tTrainRed, tTestRed = train_test_split(xRed, tRed, test_size=0.3)

#scaling the red wine dataset
scaler.fit(xTrainRed)
xTrainRed = scaler.transform(xTrainRed)
xTestRed = scaler.transform(xTestRed)

rfr = RandomForestRegressor(n_estimators=100,criterion="mse",bootstrap=True,random_state=42,verbose=0,min_samples_split=2,max_depth=1000)

#Training the White wine data
rfr.fit(xTrainWhite, tTrainWhite)
rfr.fit(xTrainWhite, tTrainWhite)

#Training the Red wine data
rfr.fit(xTrainRed, tTrainRed)

yTrainWhite = rfr.predict(xTrainWhite)
yTestWhite = rfr.predict(xTestWhite)

yTrainRed = rfr.predict(xTrainRed)
yTestRed = rfr.predict(xTestRed)

#MAE,MSE & RMSE for White wine dataset
maeTestWhite = mean_absolute_error(tTestWhite, yTestWhite)
maeTrainWhite = mean_absolute_error(tTrainWhite, yTrainWhite)

mseTestWhite = mean_squared_error(tTestWhite, yTestWhite)
mseTrainWhite=  mean_squared_error(tTrainWhite, yTrainWhite)

rmseTestWhite = mean_squared_error(tTestWhite, yTestWhite, squared=False)
rmseTrainWhite = mean_squared_error(tTrainWhite, yTrainWhite, squared=False)

rsquaredscoreWhite=r2_score(tTestWhite, yTestWhite)

print('\n')

print("Mean absolute error for TEST formula is:", maeTestWhite)
print("Mean absolute error for TRAIN formula is:", maeTrainWhite)
print("Mean squared error for TEST formula is:", mseTestWhite)
print("Mean squared error for TRAIN formula is:", mseTrainWhite)
print("RMSE for TEST formula", rmseTestWhite)
print("RMSE for TRAIN formula", rmseTrainWhite)
print("R2 score for WHITE wine is: ", rsquaredscoreWhite)

print('===================================================================')

#MAE, MSE & RMSE for Red wine dataset

maeTestRed = mean_absolute_error(tTestRed, yTestRed)
maeTrainRed = mean_absolute_error(tTrainRed, yTrainRed)

mseTestRed = mean_squared_error(tTestRed, yTestRed)

rmseTestRed = mean_squared_error(tTestRed, yTestRed, squared=False)
rmseTrainRed = mean_squared_error(tTrainRed, yTrainRed, squared=False)

rsquaredscoreRed=r2_score(tTestRed, yTestRed)

print("Mean absolute error for TEST formula is:", maeTestRed)
print("Mean absolute error for TRAIN formula is:", maeTrainRed)
print("Mean squared error for TEST formula is:", mseTestRed)
print("RMSE for TEST formula", rmseTestRed)
print("RMSE for TRAIN formula", rmseTrainRed)
print("R2 score for RED wine is: ", rsquaredscoreRed)


#plot the trees 
# dotfile = six.StringIO()
# i_tree = 0
# for tree_in_forest in rfr.estimators_:
#     with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
#         my_file = tree.plot_tree(tree_in_forest,
#                                        feature_names=xWhite.columns,
#                                        filled=True,
#                                        rounded=True)
#         # os.system('tree_' + str(i_tree) + '.dot'+' -o tree_' + str(i_tree) + '.png')
#         os.system('dot -Tpng tree.dot -o tree.png')
#         i_tree = i_tree + 1