# src/prediction.py

import numpy as np
import pandas as pd
import os
from src.model import load_model_only  # Use the new loader

# Constants
MODEL_PATH = os.path.join('models', 'my_model.h5')
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Load model once
model = load_model_only(MODEL_PATH)

def predict_heart_disease(input_dict):
    """
    Predicts heart disease risk based on input features.

    Parameters:
        input_dict (dict): A dictionary containing the 13 heart disease features.

    Returns:
        dict: Prediction label and confidence score.
    """
    if not all(feature in input_dict for feature in FEATURES):
        raise ValueError("Missing one or more required features")

    # Convert input to DataFrame without scaling
    input_df = pd.DataFrame([input_dict], columns=FEATURES)

    # Predict directly (ensure input matches model's expected format)
    prediction = model.predict(input_df)[0][0]
    label = int(prediction > 0.5)

    return {
        'prediction': label,
        'confidence': float(prediction)
    }