# Name: Kofi Boateng Index_number: 10022200200
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def run():
    st.subheader("🧩 K-Means Clustering")

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Dataset:")
        st.dataframe(df.head())

        numeric_df = df.select_dtypes(include=["float64", "int64"])
        st.write("Using the following numeric columns for clustering:")
        st.write(numeric_df.columns.tolist())

        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numeric columns to visualize clusters.")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        num_clusters = st.slider("Select number of clusters", 2, 10, 3)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(X_scaled)

        df['Cluster'] = clusters

        st.success(f"Clustering completed with {num_clusters} clusters!")

        if numeric_df.shape[1] == 2:
            fig = px.scatter(df, x=numeric_df.columns[0], y=numeric_df.columns[1], color="Cluster", title="2D Cluster Plot")
        else:
            fig = px.scatter_3d(df, x=numeric_df.columns[0], y=numeric_df.columns[1], z=numeric_df.columns[2], color="Cluster", title="3D Cluster Plot")
        
        st.plotly_chart(fig)

        st.download_button("📥 Download Clustered Data as CSV", data=df.to_csv(index=False), file_name="clustered_data.csv", mime="text/csv")
