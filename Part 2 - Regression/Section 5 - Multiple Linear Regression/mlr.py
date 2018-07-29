# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 00:10:53 2018

@author: Mohammad Doosti Lakhani
"""

# imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# imporing dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# encoding categorial data types to labelEncoder and onehotencoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
labelencoder_x = labelencoder_x.fit(x[:,3])
x[:,3] = labelencoder_x.transform(x[:,3])

onehotencoder_x = OneHotEncoder(categorical_features=[3])
onehotencoder_x = onehotencoder_x.fit(x)
x = onehotencoder_x.transform(x).toarray()

# getting rid of Dummy Variables Trap
x = x [:,1:]

# splitting dataset into Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.8 , random_state=0)

# optimal model using Backward Elimination approach
import statsmodels.formula.api as sm
x_train = np.append(arr = np.ones(shape= (40,1)) , values = x_train ,axis = 1) # adding Constant value
x_test = np.append(arr = np.ones(shape= (10,1)) , values = x_test ,axis = 1)
x_optimal = x_train
regressor_OLS = sm.OLS(endog = y_train,exog = x_optimal).fit()
print(regressor_OLS.summary())

# backward main loop
while np.argmax(regressor_OLS.pvalues[:],axis = 0) > 0.05:
    indv_index =  np.argmax(regressor_OLS.pvalues[:],axis = 0) # the index of independent variable with max pvalue
    x_optimal = np.delete(x_optimal,indv_index,axis =1)
    x_test = np.delete(x_test,indv_index,axis =1)
    regressor_OLS = sm.OLS(endog = y_train,exog = x_optimal).fit()
    print(regressor_OLS.summary())
        
# prediciting using optimal model    
y_predict = regressor_OLS.predict(x_test)

# visualize the fitted model and our data (3D)