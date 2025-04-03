# retrain.py
import os
import json
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.models import load_model as keras_load_model
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data_from_db, preprocess_data

# Optional: Only include matplotlib when needed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# To store latest evaluation
latest_metrics_temp = {}

def retrain_model(model_path='models/my_model.h5', epochs=1):
    global latest_metrics_temp

    print("📊 Loading data from DB...")
    df = load_data_from_db()

    print("🧼 Preprocessing data...")
    X_processed, y, _ = preprocess_data(df)

    print("🔀 Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    print("📦 Loading model...")
    model = keras_load_model(model_path)

    print("⚙️ Compiling model...")
    model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=["accuracy"])

    print("🚀 Training model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    print("✅ Training complete.")

    print("🧪 Evaluating...")
    y_pred_probs = model.predict(X_test).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    conf_mat = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "confusion_matrix": conf_mat.tolist(),
        "history": {
            "loss": [round(val, 4) for val in history.history.get("loss", [])],
            "val_loss": [round(val, 4) for val in history.history.get("val_loss", [])],
            "accuracy": [round(val, 4) for val in history.history.get("accuracy", [])],
            "val_accuracy": [round(val, 4) for val in history.history.get("val_accuracy", [])]
        }
    }

    print("📊 Saving visualizations...")
    os.makedirs("static/img", exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("static/img/confusion_matrix.png")
    plt.close()

    # Loss Curve
    plt.figure(figsize=(6, 4))
    plt.plot(metrics["history"]["loss"], label='Training Loss')
    plt.plot(metrics["history"]["val_loss"], label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("static/img/loss_curve.png")
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(6, 4))
    plt.plot(metrics["history"]["accuracy"], label='Training Accuracy')
    plt.plot(metrics["history"]["val_accuracy"], label='Validation Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("static/img/accuracy_curve.png")
    plt.close()

    latest_metrics_temp = {
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "precision": metrics["precision"],
        "recall": metrics["recall"]
    }

    return model, metrics