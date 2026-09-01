# OrthoTrack AI - Backend (Model Server)

This is the ONLY backend this project needs. No MongoDB, no Express,
no login/auth - just a small Python API that runs your trained
Random Forest models and returns correct/incorrect predictions to
the React frontend.

## Setup

1. Put your 3 trained model files into the `models/` folder:
   ```
   models/
     rf_model_shoulder.joblib
     rf_model_hip.joblib
     rf_model_knee.joblib
   ```
   (Download these from Google Drive: OrthoTrack_AI/REHAB24_Processed/)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python model_server.py
   ```
   It starts on http://localhost:8000 and must stay running while
   you use the website.

## Check it's working

Open http://localhost:8000/health in your browser. You should see:
```json
{"status": "ok", "models_loaded": ["Shoulder", "Hip", "Knee"]}
```

## Endpoint

POST /predict
```json
{
  "region": "Shoulder",
  "angles": [30.1, 31.4, ... 30 numbers total]
}
```
Returns:
```json
{
  "region": "Shoulder",
  "correct": true,
  "confidence": 0.87,
  "prob_correct": 0.87,
  "prob_incorrect": 0.13
}
```
