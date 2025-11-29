import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", filename)
    print("Loading dataset:", os.path.abspath(file_path))
    return pd.read_csv(file_path)

def preprocess_data(df):
    features = [
        "Rainfall",
        "Humidity9am",
        "Humidity3pm",
        "Temp9am",
        "Temp3pm",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Pressure9am",
        "Pressure3pm"
    ]

    df_clean = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    return X_scaled, df_clean, scaler
