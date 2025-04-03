# preprocessing.py
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

DATABASE = 'heart_data.db'
TABLE_NAME = 'heart_records'

# Load data directly from the SQLite database
def load_data_from_db():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df

# Preprocessing function based on your notebook setup
def preprocess_data(df):
    # Split features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert scaled array back to DataFrame for consistency
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled_df, y, scaler

# SMOTE function remains the same
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled