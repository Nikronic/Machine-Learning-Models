# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:46:04 2018

@author: Mohammad Doosti Lakhani
"""

"""############################## in this code, we include Gender  ########################"""

# imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,1:4].values 
y = dataset.iloc[:,-1].values

# feature scaling
from sklearn.preprocessing import StandardScaler
standardscaler_x = StandardScaler()
standardscaler_x = standardscaler_x.fit(x[:,1:3])
x[:,1:3] = standardscaler_x.transform(x[:,1:3])

# encoding categorial data (Gender) types to labelEncoder and onehotencoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
labelencoder_x = labelencoder_x.fit(x[:,0])
x[:,0] = labelencoder_x.transform(x[:,0])

onehotencoder_x = OneHotEncoder(categorical_features=[0])
onehotencoder_x = onehotencoder_x.fit(x)
x = onehotencoder_x.transform(x).toarray()

# splitting dataset into Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.75 , random_state=0)

# Fitting Logistic Regression model to train data
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver='liblinear')
classifier = classifier.fit(x_train,y_train)

# Make the prediction on train set
y_train_pred = classifier.predict(x_train)

# Make the prediction on train set
y_test_pred = classifier.predict(x_test)


# Acurracy on test and train set
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train,y_train_pred)
cm_test = confusion_matrix(y_test,y_test_pred)

import os
import sys

scriptpath = "../../Tools" # functions of acc and CAP
# Add the directory containing your module to the Python path
sys.path.append(os.path.abspath(scriptpath))
import accuracy as ac

t_train,f_train,acc_train = ac.accuracy_on_cm(cm_train)
print('Train status = #{} True, #{} False, %{} Accuracy'.format(t_train,f_train,acc_train*100))

t_test,f_test,acc_test = ac.accuracy_on_cm(cm_test)
print('Test status = #{} True, #{} False, %{} Accuracy'.format(t_test,f_test,acc_test*100))
      
      # For educational purpose, I didn't added PCA to reduce dimensions to plot. But if you
      #     like to do this, see the Part 9 and use lr.py visualization code to plot.