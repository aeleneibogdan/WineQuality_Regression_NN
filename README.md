For this work I will analyze the Wine Quality Data Set found on the UCI website using a Regression Algorithm.
Link: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

Data set information:
	The two datasets are related to red and white variants of the Portuguese “Vinho Verde” wine. Our goal is to implement a neural network to determine the quality of a wine, based on the physicochemical tests which have the results in the two data sets. We will process these datasets for our Neural Network.

The dataset:
Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
Output variable (based on sensory data):
12 - quality (score between 0 and 10)

For this Regression task  I have used two DATA files: winequality-white.csv and winequality-red.csv and I read them with the read_csv function from pandas library.
In total,  in each dataset there are 12 attributes that are playing a key role in analyzing the quality of the wines. The most important column is the output one that indicates the quality of the wine .
The white wine data set contains 4898 instances, while the red wine data set contains only 1599 instances, each representing a feature of the wine based on the  physicochemical tests.

We will use two types of algorithms: Multilayer Perceptron Regressor and Random Forest Regressor
