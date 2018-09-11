# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 20:26:28 2018

@author: Mohammad Doosti Lakhani
"""



# import libraries
import numpy as np
import pandas as pd
import os
import sys

scriptpath = "../../Tools" # functions of acc and CAP
# Add the directory containing your module to the Python path
sys.path.append(os.path.abspath(scriptpath))



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


# ref = https://github.com/ymoch/apyori/blob/master/apyori.py
from apyori import apriori 
rules = apriori(transactions= transactions, min_support= 0.003, min_confidence= 0.2, in_lift = 3, min_length = 2)
result = list(rules)

# Train model using Apiori algorithm 
# ref = https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df_sets = apriori(df_x, min_support=0.005, use_colnames=True)
df_rules = association_rules(df_sets,metric='lift',min_threshold= 3)
