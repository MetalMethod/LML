"""
Created on Sun Dec 17 22:11:05 2017
Igor Busquets LML
"""

#K-Means CLustering

#Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Mall_Customers.csv')
#array of required variables for analysis
#X = matrix of features
X = dataset.iloc[:, [3,4]].values

#Still dont know how many K clusters
#Using the Elbow method to find the optimal k number
from sklearn.cluster import KMeans
wcss = []
#compute 10 default clusters

for i in range(1,11):

    #params: init = method can be random, but here it avoids the inittialization trap
    #n_init = the number of times the algo runs with initial centroids
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    # fit kmeans do data X
    kmeans.fit(X)    
    # compute whitin clusters and append to wcss
    # inertia is witihn clusters sum of squares 
    wcss.append(kmeans.inertia_)

#Visualizing the Elbow method graph
plt.plot(range(1,11), wcss)
plt.title('Elbow method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

#So the optimal number of clusters is 5
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
#apply kmeans to dataset
#each observation return each cluster it belongs to
#y_kmeans is a vector with each predicted cluster
y_kmeans = kmeans.fit_predict(X)

######Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'TARGET')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')

#plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1] , s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# reset all variables
#%reset -f

