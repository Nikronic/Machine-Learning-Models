# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 00:31:44 2018

@author: Mohammad Doosti Lakhani
"""
# imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# imporing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

# Becasue of the data in dataset which is prepared to be polynominal, we do not have any splits.

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_lin_regressor = PolynomialFeatures(degree = 4)
poly_lin_regressor = poly_lin_regressor.fit(x.reshape(-1,1))
x = poly_lin_regressor.transform(x.reshape(-1,1)) # prepared x for train

# poly_lin_regressor = poly_lin_regressor.fit(x,y)
lin_regressor = LinearRegression()
lin_regressor = lin_regressor.fit(x,y)

# predicting Y values using LinearRegression with input by PolynomialFeatures 
y_prediction = lin_regressor.predict(x)

# visualize the fitted model and our data
plt.scatter(x[:,1],y, color ='red', alpha=0.6)
plt.plot(x[:,1],y_prediction,color='green')
plt.title('Level vs Salary (train data)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()
