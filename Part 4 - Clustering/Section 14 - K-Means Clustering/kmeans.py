# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 18:13:51 2018

@author: Mohammad Doosti Lakhani
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

# Finding the best value of CLUSTER_COUNT using Elbow method
from sklearn.cluster import KMeans
cluster_count_test = 10
wcss = []

for i in range(1,cluster_count_test+1):
    kmeans = KMeans(n_clusters=i, init='k-means++',n_init=50,max_iter=300,random_state=0)
    kmeans = kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plot wccs values wrt number of clusters
plt.plot(np.arange(1,11),wcss)
plt.title('ELbow of KMeans')
plt.xlabel('Number of Clusters')
plt.ylabel('WCCS value')
plt.show()

# Applying kmeans with optimal number of clusters gained by Elbow method
cluster_count = 5
kmeans = KMeans(n_clusters=cluster_count, init='k-means++',n_init=50,max_iter=300,random_state=0)
kmeans = kmeans.fit(x)

y_pred = kmeans.predict(x) # predicted labels
centroids = kmeans.cluster_centers_


# Plot scatter of datapoints with their clusters
plt.scatter(x[y_pred==0,0],x[y_pred==0,1],s=100, c='yellow',label = 'Cluster 1')
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],s=100, c='blue',label = 'Cluster 2')
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],s=100, c='purple',label = 'Cluster 3')
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],s=100, c='cyan',label = 'Cluster 4')
plt.scatter(x[y_pred==4,0],x[y_pred==4,1],s=100, c='green',label = 'Cluster 5')
plt.scatter(centroids[:,0],centroids[:,1],s=200,c='red',marker= 'd',label = 'Centroids')
plt.title('Clusters with wccs='+str(wcss[4]))
plt.xlabel('Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

    