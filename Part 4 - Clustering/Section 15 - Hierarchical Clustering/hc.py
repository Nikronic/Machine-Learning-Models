# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 19:59:37 2018

@author: Mohammad Doosti Lakhani
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values


# Finding the optimal count of clusters using Dendogram method
import scipy.cluster.hierarchy as h
dendogram = h.dendrogram(Z=h.linkage(x,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

# Fitting Hierachial clustring using optimal number of clusters
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_pred = hc.fit_predict(x)

# Plot scatter of datapoints with their clusters
plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100, c='yellow',label = 'Cluster 1')
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100, c='blue',label = 'Cluster 2')
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100, c='purple',label = 'Cluster 3')
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],s=100, c='cyan',label = 'Cluster 4')
plt.scatter(x[y_pred==4,0],x[y_pred==4,1],s=100, c='green',label = 'Cluster 5')
plt.title('Clusters (Hierachial))')
plt.xlabel('Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()