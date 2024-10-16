import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Streamlit UI
st.title("Iris Flower Clustering using K-Means and PCA")
st.write("This project demonstrates the application of K-Means clustering on the Iris dataset with PCA for dimensionality reduction.")

# Display dataset
st.subheader("Iris Dataset")
st.dataframe(df)

# Elbow method to find the optimal number of clusters
def elbow_method(data):
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    return inertia

# Data Preprocessing
X = df.iloc[:, :-1]  # features (sepal and petal measurements)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Display elbow method results
inertia = elbow_method(X_scaled)

st.subheader("Elbow Method for Optimal Clusters")
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, 'bo-')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
st.pyplot(plt)

# Choose optimal number of clusters (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)
df['Cluster'] = clusters

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA and clusters
st.subheader("K-Means Clustering with PCA")
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', label='Clusters')
centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, alpha=0.75, label='Centroids')
plt.title('Iris Data Clustering (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(*scatter.legend_elements(), title="Clusters")
st.pyplot(fig)

# Display the clustered dataset
st.subheader("Clustered Iris Dataset")
st.dataframe(df)

# Download option for clustered data
st.subheader("Download Clustered Data")
csv = df.to_csv(index=False)
st.download_button(label="Download CSV", data=csv, file_name='iris_clusters.csv', mime='text/csv')

# Summary
st.write("In this project, we applied K-Means clustering to classify Iris flowers based on their features, and used PCA to visualize the clusters. We used the elbow method to find the optimal number of clusters (k=3). The centroids of each cluster were also highlighted in the PCA plot.")

