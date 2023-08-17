# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 20:03:40 2023

@author: tomis
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples
import warnings
warnings.filterwarnings("ignore")

#%%
wine = pd.read_csv(r"D:\TechStuffs\Code\Machine Learning\Fundamentals of ML Class\Homeworks\Homework 5\wines.csv")

#%%
X = wine.values
#%% - Scale X values
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
#%%
# 1. Do a PCA on the data. How many Eigenvalues are above 1? Plotting the 2D solution
# (projecting the data on the first 2 principal components), how much of the variance is
# explained by these two dimensions, and how would you interpret them?
pca = PCA()
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

#%%
eig_vals = pca.explained_variance_ # 5 e.vals above 1
variance_exp = pca.explained_variance_ratio_

#%%
plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='.')
plt.xlabel("Principal Component 1, Variance Exp={:.2f}%".format(variance_exp[0]*100))
plt.ylabel("Principal Component 2, Variance Exp={:.2f}%".format(variance_exp[1]*100))
plt.title("Projection of Data onto first 2 Principal Components")

#%%
plt.bar(np.linspace(1,13,13), eig_vals)
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.title("Plot of PCA eigenvalues")
plt.axhline(y = 1, color = 'r', linestyle = '--')

#%%
figure, axis = plt.subplots(1, 2, figsize=(15, 8))
axis[0].bar(np.linspace(1,13,13), pca.components_[0])
axis[0].set_title("Principal Component 1")
axis[0].axhline(y = 0.2, color = 'r', linestyle = '--')
axis[0].axhline(y = -0.2, color = 'r', linestyle = '--')
axis[0].set_xticks(ticks = np.linspace(1,13,13), labels=wine.columns, rotation=70)
axis[1].bar(np.linspace(1,13,13), pca.components_[1])
axis[1].set_title("Principal Component 2")
axis[1].axhline(y = 0.2, color = 'r', linestyle = '--')
axis[1].axhline(y = -0.2, color = 'r', linestyle = '--')
axis[1].set_xticks(ticks = np.linspace(1,13,13), labels=wine.columns, rotation=70)

figure.suptitle("Importance of Each Feature for PC1 and PC2")

#%%
# 2. Use t-SNE on the data. How does KL-divergence depend on Perplexity (vary Perplexity from
# 5 to 150)? Make sure to plot this relationship. Also, show a plot of the 2D component with
# a Perplexity of 20.

X_sne = TSNE(n_components=2, perplexity=20, n_jobs=-1).fit_transform(X_scaled)

plt.scatter(X_sne[:, 0], X_sne[:, 1], marker='.')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Plot of t-SNE Components w/ Perplexity = 20")

#%%
perplexity = np.linspace(5,150,146)
perp_kl_df = pd.DataFrame(columns=perplexity, index=['KL Divergence'])

#%%
for perp in perplexity:
    print("perp:", perp)
    t_sne = TSNE(n_components=2, perplexity=perp, n_jobs=-1).fit(X_scaled)
    perp_kl_df[perp] = t_sne.kl_divergence_
    
#%%
plt.plot(perplexity, perp_kl_df.T['KL Divergence'])
plt.xlabel('Perplexity')
plt.ylabel("KL Divergence")
plt.title("KL Divergence Against Perplexity for t-SNE")

#%%
# 3. Use MDS on the data. Try a 2-dimensional embedding. What is the resulting stress of this
# embedding? Also, plot this solution and comment on how it compares to t-SNE.

# metric = False, stress = 0.3 (normalized)
# metric = True, stress = ~22k (unnormalized)
# (but same plot, dissimilarity matrices)

# If True, perform metric MDS; otherwise, perform nonmetric MDS. 
# When False (i.e. non-metric MDS), dissimilarities with 0 are considered as missing values

# fomrula discussed in class matches metric version

# If normalized_stress=True, and metric=False returns Stress-1. 
# A value of 0 indicates “perfect” fit, 0.025 excellent, 0.05 good, 0.1 fair, and 0.2 poor 

mds = MDS(n_components=2, metric=False, normalized_stress='auto')
X_mds = mds.fit_transform(X_scaled)

print("MDS stress is", mds.stress_)

plt.scatter(X_mds[:, 0], X_mds[:, 1], marker='.')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("MDS Projection onto 2 Components, Stress = {:.2f}".format(mds.stress_))

#%%
# 4. Building on one of the dimensionality reduction methods above that yielded a 2D solution
# (1-3, your choice), use the Silhouette method to determine the optimal number of clusters
# and then use kMeans with that number (k) to produce a plot that represents each wine as
# a dot in a 2D space in the color of its cluster. What is the total sum of the distance of all
# points to their respective clusters centers, of this solution?

# build on TSNE, use silhouette sum
k_values = np.linspace(2,10,9)
k_silhouete_df = pd.DataFrame(columns=k_values, index=['Silhouette Sum'])

for k in k_values:
    kmeans = KMeans(int(k), n_init='auto')
    labels = kmeans.fit_predict(X_sne)
    silhouette_scores = silhouette_samples(X_sne, labels)
    sum_silhouette_scores = sum(silhouette_scores)
    k_silhouete_df[k] = sum_silhouette_scores
    
#%%
# plot silhouette sum graph
plt.plot(k_values, k_silhouete_df.T['Silhouette Sum'])
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Silhouette Scores")
plt.xticks(k_values)

#%%
# plot colored graph
kmeans = KMeans(3, n_init='auto')
kmeans.fit(X_sne)
labels = kmeans.fit_predict(X_sne)

#%%
plot_labels = np.unique(labels)
  
for i in plot_labels:
    plt.scatter(X_sne[labels == i , 0] , X_sne[labels == i , 1] , label = "Class " + str(i+1), marker='.')
plt.legend()
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Optimal Kmeans Clustering, Silhouette Sum = {:.2f}".format(max(k_silhouete_df.iloc[0, :])))
plt.show()

#%%
# 5. Building on one of the dimensionality reduction methods above that yielded a 2D solution
# (1-3, your choice), use dBScan to produce a plot that represents each wine as a dot in a 2D
# space in the color of its cluster. Make sure to suitably pick the radius of the perimeter
# (“epsilon”) and the minimal number of points within the perimeter to form a cluster
# (“minPoints”) and comment on your choice of these two hyperparameters.

# increasing minSamples creates spiral like things, then eventually put less classes
# epsilon only showed good separation for specific value

dbscan = DBSCAN(eps=5, min_samples=15)
dbscan.fit(X_sne)
labels = dbscan.labels_
plot_labels = np.unique(labels)
  
for i in plot_labels:
    plt.scatter(X_sne[labels == i , 0] , X_sne[labels == i , 1] , label = "Class " + str(i+1), marker='.')
plt.legend()
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("CLustering using DBSCAN, Epsilon=5, MinPoints=15")
plt.show()

#%%
# Extra credit:
# a) Given your answers to all of these questions taken together, how many different kinds of wine
# do you think there are and how do they differ?


#%%
# extract datapoints from 2, 1 & 0
# do some data analysis
labels = kmeans.fit_predict(X_sne)

wine_1 = wine.iloc[list(np.where(labels == 0)[0]), :]
wine_2 = wine.iloc[list(np.where(labels == 1)[0]), :]
wine_3 = wine.iloc[list(np.where(labels == 2)[0]), :]

wine_1_info = wine_1.describe().loc[['mean'], :]
wine_2_info = wine_2.describe().loc[['mean'], :]
wine_3_info = wine_3.describe().loc[['mean'], :]

#%%
# inspect and select features with differences to plot
features = ['Malic_Acid', 'Ash_Alkalinity', 'Magnesium', 'Total_Phenols', 'Flavonoids', 'Stilbenes', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']
wine_1_info = wine_1_info[features]
wine_2_info = wine_2_info[features]
wine_3_info = wine_3_info[features]

smaller_features = ['Malic_Acid', 'Ash_Alkalinity', 'Total_Phenols', 'Flavonoids', 'Stilbenes', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280']
# Proline and Magnesium should go on different graphs because of scale

#%%

N = 9
ind = np.arange(N) 
width = 0.25

bar1 = plt.bar(ind, wine_1_info[smaller_features].values[0], width, color = 'r')
bar2 = plt.bar(ind+width, wine_2_info[smaller_features].values[0], width, color='g')
bar3 = plt.bar(ind+width*2, wine_3_info[smaller_features].values[0], width, color = 'b')

plt.xticks(ind+width, smaller_features, rotation=70)
plt.legend( (bar1, bar2, bar3), ('Class 1', 'Class 2', 'Class 3') )
plt.title("Mean Feature Values for each Class")

#%%
bigger_features_1 = ['Magnesium']#, 'Proline']

N = 1
ind = np.arange(N) 
width = 0.25

bar1 = plt.bar(ind, wine_1_info[bigger_features_1].values[0], width, color = 'r')
bar2 = plt.bar(ind+width, wine_2_info[bigger_features_1].values[0], width, color='g')
bar3 = plt.bar(ind+width*2, wine_3_info[bigger_features_1].values[0], width, color = 'b')

plt.xticks(ind+width, bigger_features_1)
plt.ylim(80, 110)
plt.legend( (bar1, bar2, bar3), ('Class 1', 'Class 2', 'Class 3') )
plt.title("Mean Magnesium Values for each Class")
#%%

bigger_features_1 = [ 'Proline']

N = 1
ind = np.arange(N) 
width = 0.25

bar1 = plt.bar(ind, wine_1_info[bigger_features_1].values[0], width, color = 'r')
bar2 = plt.bar(ind+width, wine_2_info[bigger_features_1].values[0], width, color='g')
bar3 = plt.bar(ind+width*2, wine_3_info[bigger_features_1].values[0], width, color = 'b')

plt.xticks(ind+width, bigger_features_1)
plt.ylim(400, 1100)
plt.legend( (bar1, bar2, bar3), ('Class 1', 'Class 2', 'Class 3') )
plt.title("Mean Proline Values for each Class")