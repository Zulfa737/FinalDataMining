import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Supermarket", layout="wide")

st.title("📊 Dashboard Analisis Supermarket")
st.write("Aplikasi ini melakukan Eksplorasi Data, Clustering (Segmentasi), dan Regresi (Prediksi).")

# --- 1. UPLOAD / LOAD DATA ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file 'superMarket.csv'", type=["csv"])

# Fungsi untuk memproses data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File berhasil dimuat!")

    # --- 2. EKSPLORASI DATA ---
    st.header("1. Eksplorasi Data")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preview Data")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Cek Data Kosong")
        st.write(df.isnull().sum())

    st.subheader("Statistik Deskriptif")
    st.write("Ringkasan statistik (Rata-rata, Min, Max, dll):")
    st.dataframe(df[['Unit price', 'Quantity', 'Rating', 'gross income']].describe())

    # --- 3. CLUSTERING (K-MEANS) ---
    st.markdown("---")
    st.header("2. Clustering (Segmentasi Pelanggan)")
    
    # Input User untuk Jumlah Cluster
    k_value = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=5, value=3)
    
    # Proses Clustering
    features_cluster = ['Unit price', 'Quantity', 'Rating', 'gross income']
    X_cluster = df[features_cluster]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Interpretasi Cluster
    st.subheader("Interpretasi Hasil Cluster")
    col_cl1, col_cl2 = st.columns([2, 1])
    
    with col_cl1:
        st.write("**Rata-rata Fitur per Cluster (Karakteristik):**")
        cluster_profile = df.groupby('Cluster')[features_cluster].mean()
        st.dataframe(cluster_profile.style.highlight_max(axis=0, color='lightgreen'))
    
    with col_cl2:
        st.write("**Jumlah Data per Cluster:**")
        st.write(df['Cluster'].value_counts())

    # Visualisasi Clustering
    st.subheader("Visualisasi Cluster")
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x='gross income', y='Rating', hue='Cluster', palette='viridis', s=100, ax=ax_cluster)
    ax_cluster.set_title(f'Clustering dengan {k_value} Cluster')
    st.pyplot(fig_cluster)

    # --- 4. REGRESI (LINEAR REGRESSION) ---
    st.markdown("---")
    st.header("3. Regresi (Prediksi Pendapatan)")
    st.write("Memprediksi **Gross Income** berdasarkan *Unit Price* dan *Quantity*.")

    # Split Data
    X_reg = df[['Unit price', 'Quantity']]
    y_reg = df['gross income']
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Latih Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Tampilkan Hasil Metrik
    col_reg1, col_reg2, col_reg3 = st.columns(3)
    col_reg1.metric("R2 Score (Akurasi)", f"{r2:.4f}")
    col_reg2.metric("Mean Squared Error", f"{mse:.4f}")
    col_reg3.metric("Intercept", f"{model.intercept_:.2f}")

    st.write("**Koefisien (Pengaruh Variabel):**")
    coef_df = pd.DataFrame(model.coef_, X_reg.columns, columns=['Coefficient'])
    st.dataframe(coef_df)

    # Visualisasi Regresi
    st.subheader("Visualisasi: Actual vs Predicted")
    fig_reg, ax_reg = plt.subplots(figsize=(10, 5))
    ax_reg.scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax_reg.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax_reg.set_xlabel('Actual Gross Income')
    ax_reg.set_ylabel('Predicted Gross Income')
    ax_reg.set_title('Seberapa akurat prediksinya? (Titik mendekati garis merah = Akurat)')
    st.pyplot(fig_reg)

else:
    st.info("Silakan upload file CSV di sidebar sebelah kiri untuk memulai.")
