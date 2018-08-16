# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 00:17:26 2018

@author: Mohammad Doosti Lakhani
"""

# A nonlinear and non continuos regression problem.

# imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1].values
y = dataset.iloc[:,-1].values

# reshape x and y because they just have one feature
x = x.reshape(-1,1)
# array y should stay 1D. But x should reshape to 2D

# fitting Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor= regressor.fit(x,y)
regressor2 = RandomForestRegressor(n_estimators = 10, random_state = 0) # trey to show diff in number of trees
regressor2 = regressor2.fit(x,y)


# making prediction
y_pred = regressor.predict(6.5)

# visualize the fitted model and our data
# we can see average values calculated by Random Forest Regression.
# Each step is one section with its information entropy.
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(-1,1)
plt.scatter(x,y, color ='red', alpha=0.6)
plt.scatter(6.5,y_pred,color = 'blue', marker='D',alpha = 0.5)
plt.plot(x_grid,regressor.predict(x_grid),color='green' , alpha= 0.7)
plt.plot(x_grid,regressor2.predict(x_grid),color='purple', alpha = 0.6)
plt.title('Level vs Salary (train data) using Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()
