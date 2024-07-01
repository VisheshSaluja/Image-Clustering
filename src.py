#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.metrics import calinski_harabasz_score  

class KMeansCustom:
    def __init__(self, n_clusters=10, max_iter=300, random_state=42, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_indices = np.random.permutation(np.arange(len(X)))[:self.n_clusters]
        self.centroids = X[random_indices]

    def assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        self.labels = np.argmin(distances, axis=0)

    def update_centroids(self, X):
        new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
        if np.allclose(self.centroids, new_centroids, atol=self.tol):
            return False
        self.centroids = new_centroids
        return True

    def fit(self, X):
        self.initialize_centroids(X)
        for _ in range(self.max_iter):
            self.assign_clusters(X)
            if not self.update_centroids(X):
                break

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

data_path = 'test_data.txt'
df = pd.read_csv(data_path, header=None)
data = df.values

# Normalize data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Apply UMAP for dimensionality reduction
umap_model = UMAP(n_components=2, random_state=42)
data_umap = umap_model.fit_transform(data_normalized)

# Apply the custom K-means on the UMAP output
kmeans_custom = KMeansCustom(n_clusters=10, random_state=42)
kmeans_custom.fit(data_umap)
assignments_custom = kmeans_custom.predict(data_umap)

# Evaluate with Calinski-Harabasz Score
ch_score = calinski_harabasz_score(data_umap, assignments_custom)
print(f"Calinski-Harabasz Score with Custom KMeans on UMAP data: {ch_score}")

# Save the cluster assignments 
assignments_for_submission = assignments_custom + 1
np.savetxt("cluster_assignments.txt", assignments_for_submission, fmt='%d', delimiter=',')

print("Cluster assignments saved to 'cluster_assignments.txt'.")

# Plotting Calinski-Harabasz Index for various K 
ks = range(2, 21, 2)
ch_scores = []
for k in ks:
    kmeans = KMeansCustom(n_clusters=k, random_state=42)
    kmeans.fit(data_umap)
    labels = kmeans.predict(data_umap)
    ch_score = calinski_harabasz_score(data_umap, labels)
    ch_scores.append(ch_score)

# Plotting the Calinski-Harabasz Index
plt.figure(figsize=(10, 6))
plt.plot(ks, ch_scores, marker='o')
plt.title('Calinski-Harabasz Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Calinski-Harabasz Score')
plt.grid(True)
plt.show()


# In[ ]:




