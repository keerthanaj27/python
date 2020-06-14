# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:43:18 2019
Purpose: imports the prepared pca matrix, subsets out each set of questions and does PCA and clustering for each question 
@author: kjagadeesh
"""
# import the libraries necessary 
import pandas as pd
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# reading the pca matrix that is already prepared
matrix = pd.read_csv("b_matrix_v1.csv")

# set index as var_1
matrix.set_index('var_1', inplace = True)

# only keep the questions for business location (or bloc) in this matrix
#bloc_matrix = matrix[matrix.columns[matrix.columns.str.contains('bloc')]]

# PCA components setup for bloc
pca_bloc = PCA(n_components = 20) # let's settle on 4 according to the scree plot below
bloc_pca = pca_bloc.fit_transform(matrix)

# print summary of percentage of explained
print (sum(pca_bloc.explained_variance_ratio_))

# Plot the explained variances - to see how many variables explain most of the variance and where the drop off is
features_bloc = range(pca_bloc.n_components_)
plt.bar(features_bloc, pca_bloc.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features_bloc)

# draw a correlation matrix of all the variables in bloc matrix
bloc_correl = bloc_matrix.corr()
plt.matshow(bloc_correl) 
plt.show()


# Save components to a DataFrame
bloc_PCA_components = pd.DataFrame(bloc_pca)

# after plotting - plot only the first two variables that explain the variance - change according to what the scree plot tells 
plt.scatter(bloc_PCA_components[0], bloc_PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2') 

# do k-means clustering on the reduced PCA components 

ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters= 4)
    
    # Fit model to samples
    model.fit(bloc_PCA_components.iloc[:,:4])

    print(model.cluster_centers_) 
     
    # save new clusters 
    model_save = model.fit_predict(bloc_PCA_components)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# plots the k-mean clusters and you check the elbow point    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
   
# merge with previous dataset
# create a temporary key column in both the datasets for merging 
bloc_matrix['clust'] = model_save

# export the file to csv 
bloc_matrix.to_csv('bloc_cluster_v1.csv') 


