from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os

from cleaner import clean_data
from features import engineer_features
from model import train_and_predict
from analyzer import build_response

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            df = pd.read_csv(file)
        else:
            sample_path = os.path.join(os.path.dirname(__file__), 'sample.csv')
            if not os.path.exists(sample_path):
                return jsonify({"error": "No file uploaded and sample.csv not found."}), 400
            df = pd.read_csv(sample_path)
            
        df, report = clean_data(df)
        df = engineer_features(df)
        df, metrics = train_and_predict(df)
        response_json = build_response(df, report, metrics)
        
        return jsonify(response_json)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5050)
