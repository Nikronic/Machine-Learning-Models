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
plt.scatter(x,y, color ='red', alpha=0.6)
plt.scatter(6.5,y_pred,color = 'blue', marker='D')
plt.plot(x,regressor.predict(x),color='green')
plt.title('Level vs Salary (train data) using Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()


##################################### WITH FEARURE SCALING ########################################

