import os
import json
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify , render_template
from src.retrain import retrain_model , latest_metrics_temp
from src.prediction import predict_heart_disease



app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates'))

DATABASE = 'heart_data.db'
TABLE_NAME = 'heart_records'

# Function to create database and table if not exist
def create_db_and_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            target INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# Route to serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON sent from JS
        prediction = predict_heart_disease(data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Route to handle CSV upload and insert into database
@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)

        # Check if all required columns exist
        expected_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        if not all(col in df.columns for col in expected_columns):
            return jsonify({'error': 'CSV does not contain the expected columns'}), 400

        conn = sqlite3.connect(DATABASE)
        df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
        conn.close()
        return jsonify({'message': 'Data uploaded successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/last-rows', methods=['GET'])
def get_last_rows():
    try:
        conn = sqlite3.connect(DATABASE)

        df = pd.read_sql_query(f'''
            SELECT * FROM {TABLE_NAME}
            ORDER BY rowid DESC
            LIMIT 15
        ''', conn)
        df = df.iloc[::-1]

        count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

        conn.close()

        return jsonify({
            "columns": df.columns.tolist(),  
            "rows": df.to_dict(orient='records'),
            "count": count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



last_trained_model = None


@app.route('/retrain-model', methods=['POST'])
def retrain_route():
    global last_trained_model
    try:
        model, metrics = retrain_model()
        last_trained_model = model  

        final_metrics = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "confusion_matrix": metrics["confusion_matrix"],
            "loss": metrics["history"]["loss"][-1] if metrics["history"]["loss"] else None,
            "val_loss": metrics["history"]["val_loss"][-1] if metrics["history"]["val_loss"] else None,
            "val_accuracy": metrics["history"]["val_accuracy"][-1] if metrics["history"]["val_accuracy"] else None
        }

        return jsonify({
            "message": "Model retrained successfully.",
            "metrics": final_metrics,
            "confusion_plot": "/static/img/confusion_matrix.png",
            "loss_plot": "/static/img/loss_curve.png",
            "accuracy_plot": "/static/img/accuracy_curve.png"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#  Route to save the last trained model

@app.route('/save-model', methods=['POST'])
def save_model():
    global last_trained_model
    try:
        if last_trained_model:
            os.makedirs("models", exist_ok=True)
            last_trained_model.save('models/my_model.h5')

            # ✅ Save metrics
            from src.retrain import latest_metrics_temp
            if latest_metrics_temp:
                with open("models/latest_metrics.json", "w") as f:
                    json.dump(latest_metrics_temp, f)

            return jsonify({"message": "✅ Model and metrics saved."}), 200
        else:
            return jsonify({"error": "No retrained model available."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/evaluate-model', methods=['POST'])
def evaluate_model():
    try:
        metrics_path = 'models/latest_metrics.json'
        plot_path = '/static/img/confusion_matrix.png'

        if not os.path.exists(metrics_path):
            return jsonify({"error": "No saved metrics found."}), 404

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        return jsonify({
            "metrics": metrics,
            "confusion_plot": plot_path
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500






if __name__ == '__main__':
    create_db_and_table()
    app.run(debug=True)

