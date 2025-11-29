import os
import pickle
from preprocessing import load_data, preprocess_data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("\n=== TRAINING MODEL K-MEANS DENGAN 9 FITUR CUACA ===")

# 1. Load dataset
df = load_data("weatherAUS.csv")

# 2. Preprocess (scale + clean)
X_scaled, df_clean, scaler = preprocess_data(df)

print(f"Data setelah cleaning: {len(df_clean)} baris")

# 3. Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

print("Training selesai.")
print("Inertia:", kmeans.inertia_)

# 4. Evaluasi
score = silhouette_score(X_scaled, kmeans.labels_)
print("Silhouette Score:", score)

# 5. Save model + scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "..", "model")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(kmeans, f)

with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("\nModel disimpan di folder model/")
