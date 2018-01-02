"""
Igor Busquets LML
"""

#REINFORCEMENT LEARNING: Thompsom Sampling
# tries to estimate the highest probability of success

#importing depedencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
###########Importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# CHECK Upper Confidence Bound algo for the random implementation

######### Implementing Thompsom Sampling
N = 10000
# d = number of elements
d = 10 
ads_selected = []  

#vector of d elements, the number of each time a Ad is rated 1 (selected) AND rated 0 (not selected)
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0  = [0] * d

total_reward = 0

for n in range(0, N):
    ad = 0
    #max random  draw, the max of a selected Ad by random selections
    max_random = 0
    for i in range(0, d):
        #random_beta  = different random draws 
        #foloows the math formula of slie 2
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
                
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    
    ads_selected.append(ad)
    reward = dataset.values[n,  ad]
    
    #increment the rewards
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1 
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1 
    # total_reward  = measure of the algorith (almotst 25% better than UCB, more than a double of random selection)
    total_reward = total_reward + reward
        
#####VIsulalising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each Ad was selected')
plt.show()