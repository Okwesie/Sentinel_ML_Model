"""
AIoT Passenger Safety System - Flask API V2
Updated for Hybrid Ensemble (XGBoost + Rules) with Auto-Cloud Loading
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import gdown
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

# ============================================================================
# CLOUD MODEL LOADING
# ============================================================================

GDRIVE_FILE_ID = '1svIVO8BkrPSpHezqmhEXBcQjpGug3nCD'

def ensure_model_exists():
    base_path = os.path.dirname(__file__)
    model_dir = os.path.join(base_path, 'models')
    model_path = os.path.join(model_dir, 'ghana_safety_ensemble_v2.pkl')
    
    if not os.path.exists(model_path):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print("☁️ Model missing. Downloading from Google Drive...")
        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            print(f"❌ Download failed: {e}")
    return model_path

def load_ensemble():
    try:
        model_path = ensure_model_exists()
        if not os.path.exists(model_path):
            return None
        
        package = joblib.load(model_path)
        print(f"✅ Ensemble Loaded: v{package.get('version', '2.1.0_hybrid')}")
        return package
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return None

PACKAGE = load_ensemble()

# ============================================================================
# HYBRID LOGIC
# ============================================================================

def get_hybrid_prediction(telemetry):
    if not PACKAGE:
        return {"error": "Model package not initialized"}

    # 1. ML Probability (XGBoost)
    model = PACKAGE['models']['xgb']
    scaler = PACKAGE['scaler']
    features = PACKAGE['feature_names']
    thresholds = PACKAGE.get('thresholds', {'medium': 0.3, 'high': 0.6})

    input_df = pd.DataFrame([telemetry])
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    X_scaled = scaler.transform(input_df[features])
    ml_prob = float(model.predict_proba(X_scaled)[0][1])

    # 2. Expert Rules (Ghana Context)
    rule_score = 0
    violations = []
    
    if telemetry.get('max_speed', 0) > 120:
        rule_score += 0.5
        violations.append("EXCESSIVE_SPEED")
    
    if telemetry.get('is_night', 0) == 1 and telemetry.get('stop_ratio', 0) > 0.3:
        rule_score += 0.4
        violations.append("HIGH_RISK_NIGHT_STOP")
        
    if telemetry.get('speed_std', 0) > telemetry.get('avg_speed', 0) * 0.5:
        rule_score += 0.2
        violations.append("ERRATIC_DRIVING")

    # 3. Final Hybrid Score (60% AI, 40% Rules) - Capped at 1.0
    final_score = min(1.0, (ml_prob * 0.6) + (rule_score))
    
    # Determine risk level
    if final_score >= thresholds['high']:
        risk = "HIGH"
    elif final_score >= thresholds['medium']:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        'risk_level': risk,
        'hybrid_score': round(final_score, 4),
        'ml_probability': round(ml_prob, 4),
        'rule_violations': violations,
        'trigger_emergency': risk == "HIGH",
        'timestamp': datetime.now().isoformat()
    }

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/health')
def health():
    return jsonify({
        'status': 'operational',
        'model_v2_loaded': PACKAGE is not None,
        'engine': 'Hybrid XGBoost-Rules Ensemble',
        'sync': 'Cloud GDrive'
    })

@app.route('/api/v2/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No telemetry data provided'}), 400
        result = get_hybrid_prediction(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v2/batch', methods=['POST'])
def batch():
    try:
        data = request.get_json()
        trips = data.get('trips', [])
        results = [get_hybrid_prediction(t) for t in trips]
        return jsonify({'results': results, 'count': len(results)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)