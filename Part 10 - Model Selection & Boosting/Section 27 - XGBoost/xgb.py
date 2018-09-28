# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:30:10 2018

@author: Mohammad Doosti Lakhani
"""
"""
Be sure to do not set your file name "xgboost.py". if you do, you cannot import library!!!
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

# Fitting XGboost model
from xgboost import XGBClassifier
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic','eval_metric':'auc'}
classifier = XGBClassifier(**param)
classifier = classifier.fit(x_train,y_train)

# Acurracy on test and train set using K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accs_train = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv=10)
accs_test = cross_val_score(estimator = classifier, X = x_test, y = y_test, cv=10)

acc_train = accs_train.mean()
acc_test = accs_test.mean()
print('Train status =  %{} Accuracy'.format(acc_train*100))
print('Test status =  %{} Accuracy'.format(acc_test*100))
