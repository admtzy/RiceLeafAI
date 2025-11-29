import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

# Load model & scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "model", "kmeans_model.pkl")
scaler_path = os.path.join(base_dir, "..", "model", "scaler.pkl")

with open(model_path, "rb") as f:
    kmeans = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.title("🌧️ Aplikasi Deteksi Risiko Banjir (K-Means)")
st.write("Masukkan data cuaca untuk memprediksi tingkat risiko banjir.")

# Fitur sesuai weatherAUS.csv
Rainfall = st.number_input("Curah Hujan (mm)", 0.0, 500.0)
Humidity9am = st.number_input("Kelembaban 9AM (%)", 0.0, 100.0)
Humidity3pm = st.number_input("Kelembaban 3PM (%)", 0.0, 100.0)
Temp9am = st.number_input("Suhu 9AM (°C)", -10.0, 60.0)
Temp3pm = st.number_input("Suhu 3PM (°C)", -10.0, 60.0)
WindSpeed9am = st.number_input("Kecepatan Angin 9AM (km/h)", 0.0, 150.0)
WindSpeed3pm = st.number_input("Kecepatan Angin 3PM (km/h)", 0.0, 150.0)
Pressure9am = st.number_input("Tekanan Udara 9AM (hPa)", 800.0, 1100.0)
Pressure3pm = st.number_input("Tekanan Udara 3PM (hPa)", 800.0, 1100.0)

if st.button("Prediksi Risiko"):
    # Data input sesuai urutan fitur training
    data = np.array([[
        Rainfall, 
        Humidity9am, Humidity3pm,
        Temp9am, Temp3pm,
        WindSpeed9am, WindSpeed3pm,
        Pressure9am, Pressure3pm
    ]])

    # Scaling
    data_scaled = scaler.transform(data)

    # Prediksi cluster
    cluster = kmeans.predict(data_scaled)[0]

    # Output risiko
    if cluster == 0:
        st.success("🟢 Risiko Banjir Rendah")
    elif cluster == 1:
        st.warning("🟡 Risiko Banjir Sedang")
    else:
        st.error("🔴 Risiko Banjir Tinggi")

    # ====== Visualisasi (2D PCA) ======
    st.subheader("Visualisasi Cluster (PCA 2D)")

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    transformed_centers = pca.fit_transform(kmeans.cluster_centers_)
    transformed_input = pca.transform(data_scaled)

    plt.figure(figsize=(6,5))
    # Cluster center
    for i, center in enumerate(transformed_centers):
        plt.scatter(center[0], center[1], marker='X', s=200, label=f"Cluster {i} center")
    
    # Input user
    plt.scatter(transformed_input[0][0], transformed_input[0][1], color='red', s=120, label="Input Anda")

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Visualisasi Cluster Menggunakan PCA")
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)
