# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:47:19 2018

@author: Mohammad Doosti Lakhani
"""

# Imporing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,-1].values

# Splitting dataset into Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.75 , random_state=0)

# Fitting the Naive Bayes (Gausiian form) model to the train set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0, presort=True, criterion='entropy')
classifier = classifier.fit(x_train,y_train)

# Make the prediction on train set
y_train_pred = classifier.predict(x_train)

# Make the prediction on train set
y_test_pred = classifier.predict(x_test)

# Acurracy on test and train set using K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accs_train = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv=10)
accs_test = cross_val_score(estimator = classifier, X = x_test, y = y_test, cv=10)

acc_train = accs_train.mean()
acc_test = accs_test.mean()
print('Train status =  %{} Accuracy'.format(acc_train*100))
print('Test status =  %{} Accuracy'.format(acc_test*100))
      
     
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
plt.show()