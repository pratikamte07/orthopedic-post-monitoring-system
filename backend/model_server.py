"""
model_server.py

Small Flask API that loads the 3 trained Random Forest models
(Shoulder, Hip, Knee) and exposes a /predict endpoint.

Your React app sends a 30-point angle sequence for one completed
repetition, along with which exercise region it is, and gets back
whether the model thinks it was performed correctly.

SETUP:
1. Download your 3 model files from Google Drive
   (OrthoTrack_AI/REHAB24_Processed/rf_model_shoulder.joblib, etc.)
   and put them in the same folder as this script, inside a
   subfolder called "models/".
2. pip install flask flask-cors scikit-learn joblib numpy pandas
3. python model_server.py
   -> runs on http://localhost:8000
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows your React app (different port) to call this API

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# region name (used by the frontend) -> actual saved filename
REGION_FILES = {
       "Shoulder": "rf_model_Shoulder.pkl",
       "Hip": "rf_model_Hip.pkl",
       "Knee": "rf_model_Knee.pkl",
   }
models = {}

for region, filename in REGION_FILES.items():
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        models[region] = joblib.load(path)
        print(f"Loaded model for {region}")
    else:
        print(f"WARNING: no model found for {region} at {path}")


def extract_features_from_sequence(angles):
    """
    angles: list/array of 30 floats (one repetition's resampled
    joint-angle sequence), same feature logic used in training.
    """
    angles = np.array(angles, dtype=float)

    features = {
        "min_angle": angles.min(),
        "max_angle": angles.max(),
        "range_angle": angles.max() - angles.min(),
        "mean_angle": angles.mean(),
        "std_angle": angles.std(),
    }

    velocity = np.diff(angles)
    features["mean_abs_velocity"] = np.abs(velocity).mean()
    features["max_abs_velocity"] = np.abs(velocity).max()
    features["std_velocity"] = velocity.std()

    # Must match the exact column order used during training
    ordered = ["min_angle", "max_angle", "range_angle", "mean_angle",
               "std_angle", "mean_abs_velocity", "max_abs_velocity", "std_velocity"]
    return pd.DataFrame([[features[k] for k in ordered]], columns=ordered)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    region = data.get("region")
    angles = data.get("angles")  # expects a list of 30 numbers

    if region not in models:
        return jsonify({"error": f"No model loaded for region '{region}'"}), 400

    if not angles or len(angles) < 5:
        return jsonify({"error": "angles must be a non-empty list of numbers"}), 400

    try:
        X = extract_features_from_sequence(angles)
        clf = models[region]

        prediction = int(clf.predict(X)[0])          # 1 = correct, 0 = incorrect
        proba = clf.predict_proba(X)[0]                # [P(incorrect), P(correct)]
        confidence = float(max(proba))

        return jsonify({
            "region": region,
            "correct": bool(prediction == 1),
            "confidence": round(confidence, 3),
            "prob_correct": round(float(proba[1]), 3),
            "prob_incorrect": round(float(proba[0]), 3),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(models.keys())
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
