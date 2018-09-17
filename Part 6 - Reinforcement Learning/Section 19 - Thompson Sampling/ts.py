# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 18:42:25 2018

@author: Mohammad Doosti Lakhani
"""

# import libraries
import numpy as np
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thomson Sampling
m,n = dataset.shape

ad_selected = np.zeros(m) # selected ads by users

N1 = np.zeros(n,dtype=np.float16) # ad i got reward count
N0 = np.zeros(n,dtype=np.float16) #ad i did not get reward count

total_reward = 0

# implementation in vectorized form
for i in range(0,m):
    max_index = 0
    theta = np.random.beta(N1+1,N0+1)
    max_index = np.argmax(theta)
    ad_selected[i] = max_index
    reward = dataset.values[i,max_index]
    if reward == 1: 
        N1[max_index] += 1
    else:
        N0[max_index] += 1
    total_reward+= reward