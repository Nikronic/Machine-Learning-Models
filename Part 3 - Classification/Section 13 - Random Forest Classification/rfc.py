# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 21:20:36 2018

@author: Mohammad Doosti Lakhani
"""

# Imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

scriptpath = "../../Tools" # functions of acc and CAP
# Add the directory containing your module to the Python path
sys.path.append(os.path.abspath(scriptpath))
import accuracy as ac

# Importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values


# Splitting dataset into Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.75 , random_state=0)

# Fitting the Naive Bayes (Gausiian form) model to the train set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=7,criterion='gini',random_state=0)
classifier = classifier.fit(x_train,y_train)

""" Try to uncomment below code to try different criterion algorithms
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=7,criterion='entropy',random_state=0)
classifier = classifier.fit(x_train,y_train)

"""

# Make the prediction on train set
y_train_pred = classifier.predict(x_train)

# Make the prediction on train set
y_test_pred = classifier.predict(x_test)

# Acurracy on test and train set
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train,y_train_pred)
cm_test = confusion_matrix(y_test,y_test_pred)


t_train,f_train,acc_train = ac.accuracy_on_cm(cm_train)
print('Train status = #{} True, #{} False, %{} Accuracy'.format(t_train,f_train,acc_train*100))

t_test,f_test,acc_test = ac.accuracy_on_cm(cm_test)
print('Test status = #{} True, #{} False, %{} Accuracy'.format(t_test,f_test,acc_test*100))
        
 
ac.capcurve(y_train,classifier.predict_proba(x_train),"Train")
ac.capcurve(y_test,classifier.predict_proba(x_test),"Test")
      
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 1))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier - Entropy (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 1))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier - Entropy (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()



