import os
import pickle
import pandas as pd
from preprocessing import load_data, preprocess_data
from sklearn.metrics import silhouette_score

print("\n=== EVALUASI PCA + KMEANS ===")

df = load_data("weatherAUS.csv")
X_scaled, df_clean, scaler = preprocess_data(df)

# Load model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "..", "model")

kmeans = pickle.load(open(os.path.join(model_dir, "kmeans_model.pkl"), "rb"))
pca = pickle.load(open(os.path.join(model_dir, "pca.pkl"), "rb"))

# Transform ke ruang PCA yang sama
X_pca = pca.transform(X_scaled)

clusters = kmeans.predict(X_pca)
df_clean["Cluster"] = clusters

print(df_clean.head(20))

score = silhouette_score(X_pca, clusters)
print("Silhouette Score:", score)

df_clean.to_csv(os.path.join(model_dir, "cluster_evaluation.csv"), index=False)
print("Evaluasi disimpan.")
