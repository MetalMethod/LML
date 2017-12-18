"""
Created on Mon Dec 18 20:59:05 2017 @ Mouse´s house
Igor Busquets LML
"""

# Association Rule Learning (ARL)
# Apriori Algorithm implementation

#the problem: #Optimizing the sales in a grocery store. 
#Wich products sell weel with another productos.

#Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#requires apyori.py
from apyori import apriori

#Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#header = none - first line of eader title on the dataset should be ignored

#Apriori is a special kind of dataset
# requires a list of transactions(that are also lists)
#list of transactions
transactions = []

#iterate all rows
#create a list of selected . 
# there are 20 columns in the dataset
#apriori requires a string
#single_transaction = one transaction, the many products bought by one customer
#single_transaction = [str(dataset.values[i, j]) for j in range(0, 20)]

for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

##### Training Apriori on the dataset
#from apyari import apriori
#transactions as input, rules as output
########
#min_support
#only consider products bought 3 times a day, durring 7 days of week(the time space of the dataset)
#support = 3x7/7500 = 0.003
######
#min_confidence = % of cases when the rules must be correct
#try different values
#confidence too high will get no rule, because the treshold is too high
# 80% is too much. 40% get obvious rules, 20% is ok
######
#min_lift
#try different values
#lift above 3 is good. Rules are more relevant. 
######
#min_lenght = select only the transactions with at least 2 products
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2,  min_lift = 3, min_lenght = 2)
results = list(rules)

print(results[0])
print('############################')
print(results[4])
