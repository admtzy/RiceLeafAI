import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

# Load model dan scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "model", "kmeans_model.pkl")
scaler_path = os.path.join(base_dir, "..", "model", "scaler.pkl")

with open(model_path, "rb") as f:
    kmeans = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.title("🌧️ Aplikasi Deteksi Risiko Banjir (K-Means)")

# Input Streamlit
rainfall = st.number_input("Curah Hujan (mm)", 0.0, 500.0)
humidity = st.number_input("Kelembaban (%)", 0.0, 100.0)
temperature = st.number_input("Suhu (°C)", 0.0, 50.0)
river_level = st.number_input("Ketinggian Sungai (cm)", 0.0, 500.0)

if st.button("Prediksi Risiko"):
    data = np.array([[rainfall, humidity, temperature, river_level]])
    data_scaled = scaler.transform(data)
    cluster = kmeans.predict(data_scaled)[0]

    # Tampilkan hasil prediksi
    if cluster == 0:
        st.success("🟢 Risiko Banjir Rendah")
    elif cluster == 1:
        st.warning("🟡 Risiko Banjir Sedang")
    else:
        st.error("🔴 Risiko Banjir Tinggi")

    # Plot cluster
    plt.figure(figsize=(6,5))
    # Plot semua cluster center
    for i, center in enumerate(kmeans.cluster_centers_):
        plt.scatter(center[0], center[1], marker='X', s=200, label=f"Cluster {i} center")
    
    # Plot titik input user
    plt.scatter(data_scaled[0][0], data_scaled[0][1], color='red', s=100, label="Input Anda")
    
    plt.xlabel("Curah Hujan (scaled)")
    plt.ylabel("Kelembaban (scaled)")
    plt.title("Visualisasi Cluster K-Means")
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)
