# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 12:55:49 2018

@author: Mohammad Doosti Lakhani
"""

# import libraries
import numpy as np
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',names=np.arange(1,21))

# Preprocesing data to fit model
transactions = []
for sublist in dataset.values.tolist():
    clean_sublist = [item for item in sublist if item is not np.nan]
    transactions.append(clean_sublist) # remove 'nan' values # https://github.com/rasbt/mlxtend/issues/433

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_x = pd.DataFrame(te_ary, columns=te.columns_) # encode to onehot

# Train model using Apiori algorithm 
# ref = https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df_sets = apriori(df_x, min_support=0.005, use_colnames=True)
df_rules = association_rules(df_sets,metric='support',min_threshold= 0.005,support_only=True) 
# if you use only "support", it called "ECLAT"