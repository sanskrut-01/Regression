#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:38:03 2020

@author: sanskruti
"""

# Problem Statement: Predict Salary of a potential new hire using Simple Linear Regression
# Indpendent Variable: Years of Experience
# Dependent Variable: Salary
# Dataset Size: 30, Train Test Split: 20:30
# 


# Importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Importing the dataset 
dataset = pd.read_csv('Salary_Data.csv')
# [All the rows, All the columns except last one] 
# Independent variables
X = dataset.iloc[:, :-1].values
# Dependent variable
y = dataset.iloc[:,1].values

## Handle Missing Data - No missing data
#from sklearn.impute import SimpleImputer
## (what is missing value, what to do with missing value, row or column)
#imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
#imputer.fit(X[:,1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])
#
#
## Encode Categorical Data - Inpout variables are numerical
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelEncoder_X = LabelEncoder()
#X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
## Categorical Features: index 0 -> Country
#onehotEncoder = OneHotEncoder(categorical_features = [0])
#X = onehotEncoder.fit_transform(X).toarray()
#X = X[:,1:]
## Dependent variable does not need OneHotEncoder as ML algorithm will know
#labelEncoder_y = LabelEncoder()
#y = labelEncoder_y.fit_transform(y)
#
# Split dataset into Train and Test set
from sklearn.model_selection import train_test_split
# random_size = initial seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

## Feature Scaling, important
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#X_train = sc_x.fit_transform(X_train)
#X_test = sc_x.transform(X_test)

# Simple Linear Regression, the library takes care of feature scaling
# Fitting Simple Linear Regression to the Training Set
# Machine Learning, Machine: Regressor, Learning: Learn on the training set to understand the correlation.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting on the test test
y_pred = regressor.predict(X_test)

# Evaluating Model Performance
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Visualizing Training Set Results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing Test Set Results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


















































