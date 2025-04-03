# 🩺 Heart Disease Risk Prediction App

This project is a Flask-based web application that predicts the risk of heart disease using a machine learning model. It allows users to input clinical features and receive a prediction of either "High Risk" or "Low Risk" along with a confidence score. The app also includes a retraining feature using data stored in a local SQLite database.

---

## 🚀 Live Demo

👉 [View the deployed app on Render] https://mlop-summative-frxm.onrender.com/


👉 [The video demo] https://drive.google.com/file/d/1L4ZV4tlrEsNnrDCLymxNgJvWMGlOIHXN/view?usp=sharing



---

## 🧠 Features

- Predict heart disease risk using a trained deep learning model 
- Display prediction results with confidence scores
- Retrain the model using updated data from a local SQLite database
- Visualize confusion matrix, loss, and accuracy curves
- Fetch dataset from the database
- Responsive front-end using HTML/CSS/JS and Bootstrap

---

## 🛠 Tech Stack

- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Backend:** Flask (Python)
- **Machine Learning:** TensorFlow/Keras, SMOTE, Scikit-learn
- **Database:** SQLite (`heart_data.db`)
- **Deployment:** Render

---


---

## 🖥️ Run Locally

### Prerequisites

- Python 3.7+
- pip

### Installation

```bash
git clone 
cd your-repo-name

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py




