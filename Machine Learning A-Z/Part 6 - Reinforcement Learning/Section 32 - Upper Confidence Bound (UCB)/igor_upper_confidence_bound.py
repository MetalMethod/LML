"""
Created on Thu Dec 21 19:22:20 2017 @ Mouse´s house
Igor Busquets LML
"""

#REINFORCEMENT LEARNING: Upper Confidence Bound (UCB)

#importing depedencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 
###########Importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

##Implementing Random Selection
#import random
##N = number of records
#N = 10000
##d = different Ads to display
#d = 10
#ads_selected = []
#total_reward = 0
#
#for n in range(0, N):
#    ad = random.randrange(d)
#    ads_selected.append(ad)
#    reward = dataset.values[n, ad]
#    total_reward = total_reward + reward
#    
######VIsulalising the results - Histogram
#plt.hist(ads_selected)
#plt.title('Histogram of ads selections')
#plt.xlabel('Ads')
#plt.ylabel('Number of times each Ad was selected')
#plt.show()

######## Evaluating Historgram:
#total_reward = sum of the rewards of wach rounds
# total should be compared to other algos and models for evaluation

######### Implementing UCB
N = 10000
d = 10

# step 1

ads_selected = []  
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d

#step 2
for n in range(0, N):
    #step 3
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        
        if(numbers_of_selections[i] > 0):
        #crap the videos code gives zero division erro
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            #square root and log from math formula
            # index python is 0 so add  n + 1
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            uppper_bound = average_reward + delta_i
        else:
            #1 to power of 400, very large number
            uppper_bound = 1e400
    #step 3
        #create vector of all versions of ads
        if uppper_bound > max_upper_bound:
            max_upper_bound = uppper_bound
            ad = i
            

        

        
        
        
        
        
        
        
        
        
        