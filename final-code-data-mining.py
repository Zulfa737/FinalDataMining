import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard COVID-19 Indonesia",
    page_icon="🇮🇩",
    layout="wide"
)

# Judul Utama
st.title("🇮🇩 Dashboard Analisis COVID-19: Clustering & Regresi Ensemble")
st.markdown("---")

# ==========================================
# 1. FUNGSI LOAD & RENAME DATA
# ==========================================
@st.cache_data
def load_data():
    try:
        # Load Data Asli
        df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
        
        # Dictionary Rename (Sesuai kode Colab Anda)
        kolom_baru = {
            "Date": "Tanggal", "Location ISO Code": "Lokasi_Iso", "Location": "Provinsi",
            "New Cases": "Kasus_Baru", "New Deaths": "Kematian_Baru", "New Recovered": "Sembuh_Baru",
            "New Active Cases": "Kasus_Aktif_Baru", "Total Cases": "Total_Kasus",
            "Total Deaths": "Total_Kematian", "Total Recovered": "Total_Sembuh",
            "Total Active Cases": "Total_Kasus_Aktif", "Location Level": "Level_Lokasi",
            "City or Regency": "Kota_Kabupaten", "Province": "Nama_Provinsi_Asli",
            "Country": "Negara", "Continent": "Benua", "Island": "Pulau",
            "Time Zone": "Zona_Waktu", "Special Status": "Status_Khusus",
            "Total Regencies": "Total_Kabupaten", "Total Cities": "Total_Kota",
            "Total Districts": "Total_Kecamatan", "Total Urban Villages": "Total_Kelurahan",
            "Total Rural Villages": "Total_Desa", "Area (km2)": "Luas_Km2",
            "Population": "Populasi", "Population Density": "Kepadatan_Penduduk",
            "Longitude": "Longitude", "Latitude": "Latitude",
            "New Cases per Million": "Kasus_Baru_Per_Juta",
            "Total Cases per Million": "Total_Kasus_Per_Juta",
            "New Deaths per Million": "Kematian_Baru_Per_Juta",
            "Total Deaths per Million": "Total_Kematian_Per_Juta",
            "Total Deaths per 100rb": "Total_Kematian_Per_100rb",
            "Case Fatality Rate": "Tingkat_Kematian", "Case Recovered Rate": "Tingkat_Kesembuhan",
            "Growth Factor of New Cases": "Faktor_Pertumbuhan_Kasus_Baru",
            "Growth Factor of New Deaths": "Faktor_Pertumbuhan_Kematian_Baru"
        }
        
        # Rename dan Format Tanggal
        df = df.rename(columns=kolom_baru)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%m/%d/%Y')
        return df
        
    except FileNotFoundError:
        return None

# ==========================================
# 2. LOAD & PREPROCESS
# ==========================================
df_raw = load_data()

if df_raw is not None:
    # Sidebar Info
    st.sidebar.header("ℹ️ Informasi Data")
    
    # Filter Data (Ambil Level Provinsi Saja)
    df_prov = df_raw[df_raw['Level_Lokasi'] == 'Province']
    
    if not df_prov.empty:
        # Ambil tanggal terakhir DARI DATA PROVINSI (bukan global)
        tanggal_terakhir = df_prov['Tanggal'].max()
        st.sidebar.success(f"📅 Data Tanggal: {tanggal_terakhir.date()}")
        
        # Filter DataFrame Utama
        df_analisis = df_prov[df_prov['Tanggal'] == tanggal_terakhir].copy()
        
        # Tambah Fitur Rasio
        df_analisis['Rasio_Kematian'] = df_analisis['Total_Kematian'] / df_analisis['Total_Kasus']
        df_analisis['Rasio_Kesembuhan'] = df_analisis['Total_Sembuh'] / df_analisis['Total_Kasus']
        df_analisis = df_analisis.fillna(0)
        
        # List Fitur untuk Analisis
        fitur_lengkap = ['Total_Kasus','Total_Kematian','Total_Sembuh',
                         'Kepadatan_Penduduk','Populasi','Total_Kasus_Per_Juta',
                         'Total_Kematian_Per_Juta','Rasio_Kematian','Rasio_Kesembuhan']
        
        # TABS UTAMA
        tab1, tab2, tab3 = st.tabs(["🧩 Clustering (K-Means)", "🌲 Regresi (Random Forest)", "📄 Data Mentah"])

        # ==========================================
        # TAB 1: CLUSTERING
        # ==========================================
        with tab1:
            st.header("1. Segmentasi Provinsi (Clustering)")
            st.write("Mengelompokkan provinsi berdasarkan karakteristik COVID-19.")
            
            # Input User
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                k_value = st.slider("Pilih Jumlah Cluster (K)", 2, 8, 3)
            
            # --- Proses Clustering ---
            scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(df_analisis[fitur_lengkap]), 
                                    columns=fitur_lengkap, index=df_analisis['Provinsi'])
            
            # Elbow Method (Opsional - dalam expander)
            with st.expander("Lihat Grafik Elbow (Menentukan K Optimal)"):
                wcss = []
                for i in range(1, 11):
                    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
                    km.fit(X_scaled)
                    wcss.append(km.inertia_)
                
                fig_elbow, ax_elbow = plt.subplots(figsize=(8,3))
                ax_elbow.plot(range(1, 11), wcss, 'bx-')
                ax_elbow.set_xlabel('Jumlah Cluster')
                ax_elbow.set_ylabel('WCSS')
                ax_elbow.set_title('Metode Elbow')
                st.pyplot(fig_elbow)

            # Fit K-Means Utama
            kmeans = KMeans(n_clusters=k_value, init='k-means++', max_iter=300, n_init=10, random_state=42)
            df_analisis['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # --- Visualisasi Scatter ---
            col_viz1, col_viz2 = st.columns([2, 1])
            
            with col_viz1:
                st.subheader("Peta Sebaran Cluster")
                # Pilihan Sumbu
                x_axis = st.selectbox("Sumbu X", fitur_lengkap, index=0) # Default Total Kasus
                y_axis = st.selectbox("Sumbu Y", fitur_lengkap, index=1) # Default Total Kematian
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_analisis, x=x_axis, y=y_axis, hue='Cluster', palette='viridis', s=150, ax=ax)
                
                # Label Provinsi (Agar tidak berantakan, label jika di atas rata-rata)
                mean_x = df_analisis[x_axis].mean()
                mean_y = df_analisis[y_axis].mean()
                
                for i in range(len(df_analisis)):
                    if df_analisis[x_axis].iloc[i] > mean_x or df_analisis[y_axis].iloc[i] > mean_y:
                        ax.text(df_analisis[x_axis].iloc[i], df_analisis[y_axis].iloc[i], 
                                df_analisis['Provinsi'].iloc[i], fontsize=8)
                
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col_viz2:
                st.subheader("Statistik Cluster")
                # Tampilkan rata-rata
                cluster_stats = df_analisis.groupby('Cluster')[fitur_lengkap].mean()
                st.dataframe(cluster_stats.style.highlight_max(axis=0, color='lightgreen'))
                
                st.write("**Jumlah Provinsi per Cluster:**")
                st.write(df_analisis['Cluster'].value_counts())

        # ==========================================
        # TAB 2: REGRESI (ENSEMBLE)
        # ==========================================
        with tab2:
            st.header("2. Prediksi Kematian (Ensemble Method)")
            st.write("Menggunakan **Random Forest Regressor** untuk memprediksi jumlah kematian berdasarkan fitur lain.")
            
            # Split Data
            X = df_analisis[['Total_Kasus', 'Total_Sembuh', 'Kepadatan_Penduduk', 'Populasi']]
            y = df_analisis['Total_Kematian']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Predict
            y_pred = rf_model.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Tampilan Metrik
            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric("Akurasi (R2 Score)", f"{r2:.4f}", help="Mendekati 1.0 berarti sangat akurat")
            col_met2.metric("Mean Squared Error (MSE)", f"{mse:.0f}")
            col_met3.metric("Jumlah Estimator (Trees)", "100")
            
            # Visualisasi Regresi
            col_reg1, col_reg2 = st.columns(2)
            
            with col_reg1:
                st.subheader("Prediksi vs Aktual")
                fig_reg, ax_reg = plt.subplots(figsize=(6, 4))
                ax_reg.scatter(y_test, y_pred, color='green', alpha=0.7, label='Data Test')
                
                # Garis Ideal
                min_val = min(y.min(), y_pred.min())
                max_val = max(y.max(), y_pred.max())
                ax_reg.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prediksi Sempurna')
                
                ax_reg.set_xlabel('Kematian Aktual')
                ax_reg.set_ylabel('Kematian Prediksi')
                ax_reg.legend()
                ax_reg.grid(True, alpha=0.3)
                st.pyplot(fig_reg)
                
            with col_reg2:
                st.subheader("Fitur Paling Berpengaruh")
                # Feature Importance
                importances = rf_model.feature_importances_
                feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
                
                fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                feat_imp.plot(kind='barh', color='teal', ax=ax_imp)
                ax_imp.set_title("Feature Importance (Random Forest)")
                st.pyplot(fig_imp)

        # ==========================================
        # TAB 3: DATA RAW
        # ==========================================
        with tab3:
            st.header("Data Mentah (Hasil Preprocessing)")
            st.dataframe(df_analisis)

    else:
        st.error("Data 'Province' tidak ditemukan dalam file.")
else:
    st.warning("Silakan upload file 'covid_19_indonesia_time_series_all.csv' ke folder aplikasi.")
