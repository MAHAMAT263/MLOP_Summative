import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler

# Database config
DATABASE = 'heart_data.db'
TABLE_NAME = 'heart_records'

# 📥 Load data directly from SQLite
def load_data_from_db():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df

# 🧼 Preprocess data: separate features & target, scale features
def preprocess_data(df):
    # Split features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Return as DataFrame (TensorFlow likes this format)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled_df, y, None  # Return None instead of scaler if not needed