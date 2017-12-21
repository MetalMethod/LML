"""
Created on Thu Dec 21 19:22:20 2017 @ Mouse´s house
Igor Busquets LML
"""

#REINFORCEMENT LEARNING: Upper Confidence Bound (UCB)

#importing depedencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###########Importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Random Selection
import random
#N = number of records
N = 10000
#d = different Ads to display
d = 10
ads_selected = []
total_reward = 0

for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    
#####VIsulalising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each Ad was selected')
plt.show()