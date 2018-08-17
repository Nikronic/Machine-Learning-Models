# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 23:26:31 2018

@author: Mohammad Doosti Lakhani
"""

# Imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

# Feature scaling
from sklearn.preprocessing import StandardScaler
standardscaler_x = StandardScaler()
standardscaler_x = standardscaler_x.fit(x)
x = standardscaler_x.transform(x)

# Splitting dataset into Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.75 , random_state=0)

# Fitting the K-Nearest Neighbors model to the train set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier = classifier.fit(x_train,y_train)

""" Try to uncomment below code and see the visualization output (Try different kernels)"""

"""

classifier = SVC(kernel='poly', degree = 3, random_state=0)
classifier = classifier.fit(x_train,y_train)

classifier = SVC(kernel='linear', random_state=0) ##  (Equals to SVR)
classifier = classifier.fit(x_train,y_train) 

classifier = SVC(kernel='sigmoid', random_state=0)
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

# Function for acurracy
def acc(confusion_matrix):
    t = confusion_matrix[0][0] + confusion_matrix[1][1]
    f = confusion_matrix[0][1] + confusion_matrix[1][0]
    ac = t/(t+f)
    return (t,f,ac)

t_train,f_train,acc_train = acc(cm_train)
print('Train status = #{} True, #{} False, %{} Accuracy'.format(t_train,f_train,acc_train*100))

t_test,f_test,acc_test = acc(cm_test)
print('Test status = #{} True, #{} False, %{} Accuracy'.format(t_test,f_test,acc_test*100))
      
      
      
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM - rbf kernel (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM - rbf kernel (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
