# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 04:23:21 2018

@author: Mohammad Doosti Lakhani
"""

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
y = y.reshape(-1,1)


# fitting Decision Tree Regression model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor= regressor.fit(x,y)

# making prediction
y_pred = regressor.predict(6.5)

#visualize the fitted model and our data
# we can see average values calculated by Decision Tree Regressor. 
# Each step is one section with its information entropy.
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(-1,1)
plt.scatter(x,y, color ='red', alpha=0.6)
plt.scatter(6.5,y_pred,color = 'blue', marker='D')
plt.plot(x_grid,regressor.predict(x_grid),color='green')
plt.title('Level vs Salary (train data) using Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()


##################################### WITH FEARURE SCALING ########################################

from sklearn.preprocessing import StandardScaler
standardscaler_x = StandardScaler()
x = standardscaler_x.fit_transform(x)
standardscaler_y = StandardScaler()
y = standardscaler_y.fit_transform(y)

# fitting Decision Tree Regression model
regressor2 = DecisionTreeRegressor(random_state = 0)
regressor2 = regressor2.fit(x,y)

# scaling test data for prediction
test = np.zeros(1) # we are testing just one value
test[0]= 6.5
test = test.reshape(1,1) # reshape to 2D array!
test = standardscaler_x.transform(test) # rescaling test data like train data

# making prediction
y_pred2 = regressor2.predict(test)

# inver scalingy y to real value
y_predict = standardscaler_y.inverse_transform(y_pred2)

# same value.
# but it is necesaary to scale your data. In this example dataset, it was ok!

