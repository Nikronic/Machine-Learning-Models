# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:45:13 2018

@author: Mohammad Doosti Lakhani
"""

# Importing Library
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data to one hot encoded form
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
x[:, 1] = labelencoder_country.fit_transform(x[:, 1]) # Encoding 'Geography' to numbers

labelencoder_gender = LabelEncoder()
x[:, 2] = labelencoder_gender.fit_transform(x[:, 2]) # Encoding 'Gender' to numbers

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).todense() # Encoding 'Geography' to one hot

x = x[:, 1:] # get rid of dummy variable trap

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
x= scalar.fit_transform(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Importing Library
from keras.models import Sequential
from keras.layers import Dense

# Building Model Structure
model = Sequential()
model.add(Dense(units = 22, activation='relu', input_dim = 11)) # Input layer and first hidden layer
# when we assing `input_dim = 11`, we actually creating input layer

model.add(Dense(units = 22, activation='relu')) # Second hidden layer
model.add(Dense(units = 1, activation='sigmoid')) # Output layer

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

# Fitting fully connnected NN to the Training set
model.fit(x_train, y_train, batch_size = 25, epochs = 200)

# Predicting on the Test set
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

# Get acurracy on Test set
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)

import os
import sys

scriptpath = "../../Tools" # functions of acc and CAP
# Add the directory containing your module to the Python path
sys.path.append(os.path.abspath(scriptpath))
import accuracy as ac

t_test,f_test,acc_test = ac.accuracy_on_cm(cm_test)
print('Test status = #{} True, #{} False, %{} Accuracy'.format(t_test,f_test,acc_test*100))

