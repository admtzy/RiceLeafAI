import os
import pickle
from preprocessing import load_data, preprocess_data
from sklearn.cluster import KMeans

# 1️⃣ Load data
df = load_data("weatherAUS.csv")
X = preprocess_data(df)

# 2️⃣ Load model KMeans
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "model", "kmeans_model.pkl")

# Jika model belum ada, otomatis jalankan training
if not os.path.exists(model_path):
    print("Model belum ada, jalankan train_kmeans.py terlebih dahulu.")
    exit(1)

with open(model_path, "rb") as f:
    model = pickle.load(f)

# 3️⃣ Prediksi cluster
clusters = model.predict(X)

# 4️⃣ Tampilkan hasil 20 cluster pertama
print("Cluster hasil evaluasi (20 pertama):")
print(clusters[:20])
