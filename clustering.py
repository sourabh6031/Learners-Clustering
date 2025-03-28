import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage



def plot_dendrogram(model, ax, **kwargs):
    """Function to plot dendrogram"""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plotting the dendrogram
    dendrogram(linkage_matrix, ax=ax, **kwargs)

def perform_clustering(df, method):
    """Perform clustering evaluation based on selected method"""
    
    st.write(f"### {method} Analysis")
    
    if method == "Dendrogram":
        
        sample_size = min(10000, len(df))  
        X_scaled_sample = df.sample(sample_size, random_state=42)

        # Compute linkage matrix using optimized algorithm
        linkage_matrix = linkage(X_scaled_sample, method='ward')

        # Plot dendrogram
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Optimized Hierarchical Clustering Dendrogram")
        sch.dendrogram(linkage_matrix, ax=ax, truncate_mode="level", p=3)
        st.pyplot(fig)
            
    elif method == "Silhouette Score":
        # Define range of clusters
        min_num_of_clusters = 2
        max_num_of_clusters = 21
        silhouette_scores = []
        k_values = list(range(min_num_of_clusters, max_num_of_clusters + 1))

        subset_size = min(15000, len(df))  
        subset_indices = np.random.choice(len(df), subset_size, replace=False)  # Sample indices
        X_sample = df.iloc[subset_indices]  # Extract subset

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(df)  # Fit on full dataset
            labels_sample = kmeans.labels_[subset_indices]  # Get labels for the subset
            score = silhouette_score(X_sample, labels_sample)  # Compute silhouette score on subset
            silhouette_scores.append(score)

        # Plot silhouette scores
        fig, ax = plt.subplots()
        ax.plot(k_values, silhouette_scores, marker='o', linestyle='--')
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score for Optimal k (Sampled Data)")
        ax.set_xticks(k_values)
        st.pyplot(fig)
        
    elif method == "Elbow Method":
        wcss = []
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df)
            wcss.append(kmeans.inertia_)
        
        fig, ax = plt.subplots()
        ax.plot(range(2, 10), wcss, marker='o')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("WCSS (Inertia)")
        ax.set_title("Elbow Method")
        st.pyplot(fig)
    
    elif method == "Davies Bouldin Score":
        scores = []
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(df)
            scores.append(davies_bouldin_score(df, labels))
        
        # Creating a figure and axis
        fig, ax = plt.subplots()
        ax.plot(range(2, 10), scores, marker='o')

        # Setting labels and title
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Davies Bouldin Score")
        ax.set_title("Davies Bouldin Score vs. Number of Clusters")

        # Displaying in Streamlit
        st.pyplot(fig)
