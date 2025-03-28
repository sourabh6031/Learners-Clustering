import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cluster_plot(df):
    fig, ax = plt.subplots()  # Creating a figure and axis
    sns.barplot(x=df.groupby('Cluster')['ctc'].mean().index, 
                y=df.groupby('Cluster')['ctc'].mean().values, 
                ax=ax)
    
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Average CTC")
    ax.set_title("Average CTC by Cluster Label")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    st.pyplot(fig)  # Passing the figure to Streamlit

   