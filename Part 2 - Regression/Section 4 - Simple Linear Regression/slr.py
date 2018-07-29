# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 22:37:49 2018

@author: Mohammad Doosti Lakhani
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# imporing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

# our dataset do not have any missing data or do not need to be scaled.
# splitting dataset into Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 2/3 , random_state=0)


# reshaping test and trains sets becasue they are single feature
x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# fitting Linear Regression to the train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(x_train,y_train)

# predicting test set result
y_predict = regressor.predict(x_test)
score = regressor.score(x_test,y_test) # we got 0.97 accuracy !!!

# visualize the fitted model and our data
plt.scatter(x_train,y_train, color ='red', label='Train Values', alpha=0.6)
plt.scatter(x_test,y_test,color='blue', label ='Test Values', alpha=0.6)
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title('Salary vs Exprience (train data)')
plt.xlabel('Exprience')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()
