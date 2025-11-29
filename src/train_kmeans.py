import os
import pickle
from preprocessing import load_data, preprocess_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
df = load_data("weatherAUS.csv")
features = ["Rainfall", "Humidity3pm", "Pressure9am", "Temp3pm"]
df_clean = df[features].dropna()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Latih KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Simpan model dan scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "..", "model")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(kmeans, f)

with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("Training selesai. Model dan scaler tersimpan di folder model/")
