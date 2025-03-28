import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score
from clustering import perform_clustering
from preprocess import preprocess_data, feature_engineering, outlier_treatment, feature_scaling
from clusterploting import cluster_plot 
from sklearn.decomposition import PCA
from umap import UMAP

# Streamlit App Title
st.title("Learner's Clustering App")

st.divider() # for dividing line

st.text("You can download the different uncleaned dataset to try.")
st.link_button("The link.", "https://drive.google.com/drive/folders/1dXsuLV4XXNDYr1VBwVRyHce1Nvdv_gjf?usp=sharing")


# Part 1: Dataset Selection  -------------------------------------------------------------------
dataset_option = st.radio("Choose Dataset", ["None","Use Default Dataset", "Upload My File"])
st.divider()

if dataset_option == "Use Default Dataset":
    file_id = "1BSvSIxtazgCoPGVDPhyYjMCJ9JCFT14j"
    csv_url = f"https://drive.google.com/uc?id={file_id}"

    df = pd.read_csv(csv_url)
    df = df.iloc[:,1:]
    st.write("### Preview of Default Dataset")
    st.dataframe(df.head())

elif dataset_option == "Upload My File":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.iloc[:,1:]
        st.write("### Preview of Uploaded Dataset")
        st.dataframe(df.head())
        st.divider()

        # Part 2: Checking if Data is Processed -------------------------------------------------------------------

        process_option = st.radio("Is your data already processed?", ["None","Data is Processed", "Pre-process the Data"])

        required_cols = ['job_position', 'orgyear', 'ctc_updated_year', 'ctc']

        if process_option == "Data is Processed":
            if all(col in df.columns for col in required_cols):
                st.write("Starting feature engineering...")
                df = feature_engineering(df)
                st.write("Finished feature engineering...")
                st.dataframe(df.iloc[:,1:].head())

                st.write("Feature Scaling has started...")
                X_scaled = feature_scaling(df)
                st.write("Finished feature scaling...")

                st.write("Outlier Treatment has started...")
                new_df = outlier_treatment(X_scaled)
                st.write("Finished outlier treatment...")

                st.success("Feature Engineering, Feature Scaling and Outlier Treatment done.")
            else:
                st.error("Missing required columns: " + ", ".join(set(required_cols) - set(df.columns)))
                st.stop()

        elif process_option == "Pre-process the Data":
            if all(col in df.columns for col in required_cols):
                st.write("Preprocessing has started.")
                df = preprocess_data(df)
                st.write("Preprocessing done.")

                st.write("Feature Engineering has started.")
                df = feature_engineering(df)
                st.write("feature engineering done.")
                st.dataframe(df.iloc[:,1:].head())

                st.write("Scaling has started.")
                X_scaled = feature_scaling(df)
                st.write("Scaling done.")

                st.write("Outlier Treatment has started.")
                new_df = outlier_treatment(X_scaled)
                st.success("Data Preprocessed Successfully")
            else:
                st.error("Missing required columns: " + ", ".join(set(required_cols) - set(df.columns)))
                st.stop()  

else:
    pass

st.divider()
        


# Part 3: Clustering Method Selection -------------------------------------------------------------------

check_clustering = st.checkbox("Plot graph to select 'K' ...")
if check_clustering:
    st.write("### Choose Clustering Evaluation Method")
    method = st.selectbox("Select a method:", ["None","Dendrogram", "Silhouette Score", "Elbow Method", "Davies Bouldin Score"])

    if method == 'None':
        pass
    else:
            

        # Perform Clustering Evaluation
        if dataset_option == "Use Default Dataset":
            perform_clustering(df, method)
        else:
            perform_clustering(new_df, method)
else:
    pass

st.divider()

# Step 4: Select Final "K" for Clustering -------------------------------------------------------------------
k = st.number_input("Choose Final 'K' Value. (Default=4)", min_value=2, max_value=10, value=4)

make_cluster = st.checkbox("Make Cluster")

if make_cluster:                                   # if ticked
    if dataset_option == "Use Default Dataset":    # use 'df'
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df)
        st.write("Clustering done.")

        statistics  = st.checkbox("I want to see the statistics?")    # checkbox for statistics
        if statistics:
            st.write(df.groupby("Cluster").mean())

            # to plot cluster distribution plot
            cluster_plot(df)
    else:                                           # use 'newdf'
        kmeans = KMeans(n_clusters=k, random_state=42)
        new_df['Cluster'] = kmeans.fit_predict(new_df)
        st.write("Clustering done.")

        statistics  = st.checkbox("Check this if you want to see the statistics?")    # checkbox for statistics
        if statistics:
            st.write(new_df.groupby("Cluster").mean())

            # to plot cluster distribution plot
            cluster_plot(new_df)

st.divider()

# Step 6: Visualizations UMAP-------------------------------------------------------------------
plot_umap = st.radio("Want to see UMAP Plot?", ["No", "Yes"])
st.divider()

if plot_umap == "Yes":
    st.write("### Clustering Results")
    st.text("It will take some time, be patience...")
    
    final_k = k # from kmeans

    if dataset_option == "Use Default Dataset":    # use 'df'
    # Perform PCA to reduce to 2 components
        new_df_selected = df.iloc[:,:-1]
    else:
        new_df_selected = new_df.iloc[:,:-1]

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(new_df_selected)
    df_pca = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
    df_pca['cluster'] = kmeans.labels_

    # PLOTTING UMAP
    final_k = k # from kmeans


    # Perform UMAP to reduce to 2 components
    umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_components = umap_reducer.fit_transform(df_pca)
    # Create a DataFrame with the UMAP components and cluster labels
    df_umap = pd.DataFrame(data=umap_components, columns=['UMAP1', 'UMAP2'])
    df_umap['cluster'] = kmeans.labels_



    # Plot the clusters
    plt.figure(figsize=(10, 7))
    for cluster in range(final_k):
        clustered_data = df_umap[df_umap['cluster'] == cluster]
        plt.scatter(clustered_data['UMAP1'], clustered_data['UMAP2'], label=f'Cluster{cluster}', s=50)



    plt.title('K-Means Clusters Visualized using UMAP with n_neighbors = 15 and min_dist = 0.1')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend()
    st.pyplot()

else:
    pass