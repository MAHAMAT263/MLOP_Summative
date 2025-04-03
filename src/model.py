# src/model.py

import tensorflow as tf
import os

def load_model_only(model_path='models/my_model.h5'):
    """
    Load the trained Keras model from the given path.

    Parameters:
        model_path (str): Path to the saved Keras model.

    Returns:
        tf.keras.Model: The loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model = tf.keras.models.load_model(model_path)
    return model
