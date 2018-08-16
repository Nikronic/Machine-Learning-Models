# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:14:59 2018

@author: Mohammad Doosti Lakhani
"""

"""############################## based on assignment, we do not include Gender  ########################"""


# imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

# feature scaling
from sklearn.preprocessing import StandardScaler
standardscaler_x = StandardScaler()
standardscaler_x = standardscaler_x.fit(x)
x = standardscaler_x.transform(x)

# splitting dataset into Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.75 , random_state=0)

