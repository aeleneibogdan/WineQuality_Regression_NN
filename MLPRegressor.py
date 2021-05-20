# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:27:47 2021

@author: aelen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


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

mlp = MLPRegressor(activation="relu", solver="adam", hidden_layer_sizes=(2,2),max_iter=1000,verbose=True,alpha=1e-6,random_state=42)

#Training the White wine data
mlp.fit(xTrainWhite, tTrainWhite)


#Training the Red wine data
mlp.fit(xTrainRed, tTrainRed)

yTrainWhite = mlp.predict(xTrainWhite)
yTestWhite = mlp.predict(xTestWhite)

yTrainRed = mlp.predict(xTrainRed)
yTestRed = mlp.predict(xTestRed)

#MAE,MSE & RMSE for White wine dataset
maeTestWhite = mean_absolute_error(tTestWhite, yTestWhite)
maeTrainWhite = mean_absolute_error(tTrainWhite, yTrainWhite)

mseTestWhite = mean_squared_error(tTestWhite, yTestWhite)

rmseTestWhite = mean_squared_error(tTestWhite, yTestWhite, squared=False)
rmseTrainWhite = mean_squared_error(tTrainWhite, yTrainWhite, squared=False)

rsquaredscoreWhite=r2_score(tTestWhite, yTestWhite)

print('\n')

# tTrainWhite.reshape(-1,1)
# whiteTrainAccuracy=mlp.score(tTrainWhite, yTrainWhite)
# whiteTestAccuracy=mlp.score(tTestWhite, yTestWhite)
# print("Train accuracy for the White wine dataset", whiteTrainAccuracy)
# print("Test accuracy for the White wine dataset", whiteTestAccuracy)

print("Mean absolute error for TEST formula is:", maeTestWhite)
print("Mean absolute error for TRAIN formula is:", maeTrainWhite)
print("Mean squared error for TEST formula is:", mseTestWhite)
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

# tTrainRed.reshape(-1,1)
# redTrainAccuracy=mlp.score(tTrainRed, yTrainRed)
# redTestAccuracy=mlp.score(tTestRed, yTestRed)
# print("Train accuracy for the Red wine dataset", redtrainaccuracy)
# print("Test accuracy for the Red wine dataset", redTestAccuracy)

print("Mean absolute error for TEST formula is:", maeTestRed)
print("Mean absolute error for TRAIN formula is:", maeTrainRed)
print("Mean squared error for TEST formula is:", mseTestRed)
print("RMSE for TEST formula", rmseTestRed)
print("RMSE for TRAIN formula", rmseTrainRed)
print("R2 score for RED wine is: ", rsquaredscoreRed)


plt.figure()
loss_values = mlp.loss_curve_
plt.plot(loss_values)
plt.title('Loss function')