# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 23:11:13 2018

@author: Mohammad Doosti Lakhani
"""

# imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# resolving missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis= 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# encoding categorial data types to labelEncoder and onehotencoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
labelencoder_x = labelencoder_x.fit(x[:,0])
x[:,0] = labelencoder_x.transform(x[:,0])

labelencoder_y = LabelEncoder()
labelencoder_y = labelencoder_y.fit(y)
y = labelencoder_y.transform(y)

onehotencoder_x = OneHotEncoder(categorical_features=[0])
onehotencoder_x = onehotencoder_x.fit(x)
x = onehotencoder_x.transform(x).toarray()

# splitting dataset into Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.8 , random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
standardscaler_x = StandardScaler()
standardscaler_x = standardscaler_x.fit(x_train)
x_train = standardscaler_x.transform(x_train)
x_test = standardscaler_x.transform(x_test)

