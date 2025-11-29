import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    """
    Memuat dataset CSV dari folder data/
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", filename)
    file_path = os.path.abspath(file_path)
    print("Loading:", file_path)
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Memilih fitur numerik dan menormalisasi menggunakan StandardScaler
    """
    # Fitur yang dipakai untuk clustering
    features = ["Rainfall", "Humidity3pm", "Pressure9am", "Temp3pm"]

    # Hapus baris dengan missing value
    df_clean = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    return X_scaled
