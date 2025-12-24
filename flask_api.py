"""
Simple Flask REST API for AIoT Passenger Safety System
Serves the trained ML model via REST endpoints

Run with: python flask_api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# ============================================================================
# LOAD MODEL
# ============================================================================

print("Loading model...")
try:
    model_package = joblib.load('models/aiot_detector_v1.pkl')
    MODEL = model_package['iso_forest']
    SCALER = model_package['scaler']
    FEATURE_NAMES = model_package['feature_names']
    CONTAMINATION = model_package.get('contamination', 0.05)
    print(f"✅ Model loaded successfully! Features: {len(FEATURE_NAMES)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_risk(telemetry_data):
    """
    Run prediction on telemetry data
    """
    if MODEL is None:
        return {'error': 'Model not loaded'}
    
    # Convert to DataFrame
    input_df = pd.DataFrame([telemetry_data])
    
    # Ensure all features present (fill missing with 0)
    for col in FEATURE_NAMES:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Preprocess and predict
    X_input = SCALER.transform(input_df[FEATURE_NAMES])
    ml_score = MODEL.decision_function(X_input)[0]
    prediction = MODEL.predict(X_input)[0]
    
    # Determine risk level
    if ml_score < -0.05:
        risk_level = "HIGH"
    elif ml_score < 0.05:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        'risk_level': risk_level,
        'anomaly_score': float(ml_score),
        'is_anomaly': bool(prediction == -1),
        'trigger_emergency': risk_level == "HIGH",
        'timestamp': datetime.now().isoformat(),
        'model_version': '1.0.0'
    }

def apply_rules(data):
    """
    Apply rule-based detection
    """
    alerts = []
    
    # Rule 1: Excessive speed
    if data.get('max_speed', 0) > 120:
        alerts.append({
            'rule': 'EXCESSIVE_SPEED',
            'severity': 'HIGH',
            'value': data['max_speed'],
            'threshold': 120
        })
    
    # Rule 2: Unusual stop pattern
    if data.get('stop_ratio', 0) > 0.3 and data.get('stop_count', 0) > 5:
        alerts.append({
            'rule': 'UNUSUAL_STOPS',
            'severity': 'MEDIUM',
            'value': data['stop_ratio'],
            'threshold': 0.3
        })
    
    # Rule 3: Erratic driving
    avg_speed = data.get('avg_speed', 0)
    speed_std = data.get('speed_std', 0)
    if speed_std > avg_speed * 0.5 and avg_speed > 0:
        alerts.append({
            'rule': 'ERRATIC_DRIVING',
            'severity': 'MEDIUM',
            'value': speed_std,
            'threshold': avg_speed * 0.5
        })
    
    # Rule 4: Route deviation
    if data.get('heading_variance', 0) > 90:
        alerts.append({
            'rule': 'ROUTE_DEVIATION',
            'severity': 'HIGH',
            'value': data['heading_variance'],
            'threshold': 90
        })
    
    # Rule 5: Night risk
    if data.get('is_night', 0) == 1 and data.get('stop_ratio', 0) > 0.2:
        alerts.append({
            'rule': 'NIGHT_RISK',
            'severity': 'MEDIUM',
            'value': data.get('hour_of_day', 0),
            'threshold': 0.2
        })
    
    return alerts

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """API home page with documentation"""
    return jsonify({
        'message': 'AIoT Passenger Safety API',
        'version': '1.0.0',
        'status': 'operational' if MODEL else 'model_not_loaded',
        'endpoints': {
            'GET /': 'API documentation',
            'GET /health': 'Health check',
            'POST /api/v1/predict': 'Single trip prediction',
            'POST /api/v1/batch-predict': 'Batch prediction',
            'POST /api/v1/predict-with-rules': 'Prediction with rule alerts',
            'GET /api/v1/features': 'Get required features list'
        },
        'documentation': 'http://localhost:5000/docs'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Single trip prediction endpoint
    
    Example request:
    {
        "avg_speed": 50.0,
        "max_speed": 80.0,
        "speed_std": 15.0,
        "stop_ratio": 0.2,
        "stop_count": 5,
        "heading_variance": 45.0,
        "is_night": 0,
        "hour_of_day": 14
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get prediction
        result = predict_risk(data)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/v1/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Example request:
    {
        "trips": [
            {"avg_speed": 50.0, "max_speed": 80.0, ...},
            {"avg_speed": 95.0, "max_speed": 140.0, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'trips' not in data:
            return jsonify({'error': 'No trips provided'}), 400
        
        trips = data['trips']
        results = []
        
        for i, trip in enumerate(trips):
            result = predict_risk(trip)
            result['trip_id'] = i
            results.append(result)
        
        # Calculate summary
        summary = {
            'total': len(results),
            'high_risk': sum(1 for r in results if r['risk_level'] == 'HIGH'),
            'medium_risk': sum(1 for r in results if r['risk_level'] == 'MEDIUM'),
            'low_risk': sum(1 for r in results if r['risk_level'] == 'LOW'),
            'anomalies': sum(1 for r in results if r['is_anomaly'])
        }
        
        return jsonify({
            'results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/v1/predict-with-rules', methods=['POST'])
def predict_with_rules():
    """
    Prediction with rule-based alerts
    
    Returns both ML prediction and rule violations
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get ML prediction
        ml_result = predict_risk(data)
        
        # Get rule alerts
        rule_alerts = apply_rules(data)
        
        # Combine results
        result = {
            'ml_prediction': ml_result,
            'rule_alerts': rule_alerts,
            'alert_count': len(rule_alerts),
            'high_severity_count': sum(1 for a in rule_alerts if a['severity'] == 'HIGH'),
            'recommendation': 'EMERGENCY' if ml_result['trigger_emergency'] or len([a for a in rule_alerts if a['severity'] == 'HIGH']) > 0 else 'MONITOR' if ml_result['risk_level'] == 'MEDIUM' else 'SAFE'
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/v1/features', methods=['GET'])
def get_features():
    """
    Get list of required features
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'features': FEATURE_NAMES,
        'total': len(FEATURE_NAMES),
        'example': {
            'avg_speed': 50.0,
            'max_speed': 80.0,
            'speed_std': 15.0,
            'stop_ratio': 0.2,
            'stop_count': 5,
            'heading_variance': 45.0,
            'is_night': 0,
            'hour_of_day': 14
        }
    })

@app.route('/docs')
def documentation():
    """
    API documentation page
    """
    docs = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AIoT Safety API Documentation</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 { color: #2563eb; }
            h2 { color: #1e40af; margin-top: 2rem; }
            .endpoint {
                background: #f0f9ff;
                padding: 1rem;
                margin: 1rem 0;
                border-left: 4px solid #2563eb;
                border-radius: 4px;
            }
            code {
                background: #1e293b;
                color: #e2e8f0;
                padding: 1rem;
                display: block;
                border-radius: 4px;
                overflow-x: auto;
            }
            .method {
                background: #10b981;
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-weight: bold;
                display: inline-block;
            }
            .method.post { background: #3b82f6; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 AIoT Passenger Safety API Documentation</h1>
            <p>REST API for real-time trip anomaly detection</p>
            
            <h2>Base URL</h2>
            <code>http://localhost:5000</code>
            
            <h2>Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /health</h3>
                <p>Health check endpoint</p>
                <h4>Response:</h4>
                <code>{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-24T10:30:00"
}</code>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /api/v1/predict</h3>
                <p>Single trip risk prediction</p>
                <h4>Request:</h4>
                <code>{
  "avg_speed": 50.0,
  "max_speed": 80.0,
  "speed_std": 15.0,
  "stop_ratio": 0.2,
  "stop_count": 5,
  "heading_variance": 45.0,
  "is_night": 0,
  "hour_of_day": 14
}</code>
                <h4>Response:</h4>
                <code>{
  "risk_level": "LOW",
  "anomaly_score": 0.1234,
  "is_anomaly": false,
  "trigger_emergency": false,
  "timestamp": "2025-12-24T10:30:00"
}</code>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /api/v1/batch-predict</h3>
                <p>Batch prediction for multiple trips</p>
                <h4>Request:</h4>
                <code>{
  "trips": [
    {"avg_speed": 50.0, "max_speed": 80.0, ...},
    {"avg_speed": 95.0, "max_speed": 140.0, ...}
  ]
}</code>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /api/v1/predict-with-rules</h3>
                <p>Prediction with rule-based alerts</p>
                <h4>Response includes:</h4>
                <ul>
                    <li>ML prediction results</li>
                    <li>Rule violation alerts</li>
                    <li>Combined recommendation</li>
                </ul>
            </div>
            
            <h2>Testing with cURL</h2>
            <code>curl -X POST http://localhost:5000/api/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "avg_speed": 50.0,
    "max_speed": 80.0,
    "speed_std": 15.0,
    "stop_ratio": 0.2,
    "stop_count": 5,
    "heading_variance": 45.0,
    "is_night": 0,
    "hour_of_day": 14
  }'</code>
            
            <h2>Testing with Python</h2>
            <code>import requests

data = {
    "avg_speed": 50.0,
    "max_speed": 80.0,
    "speed_std": 15.0,
    "stop_ratio": 0.2,
    "stop_count": 5,
    "heading_variance": 45.0,
    "is_night": 0,
    "hour_of_day": 14
}

response = requests.post(
    "http://localhost:5000/api/v1/predict",
    json=data
)

print(response.json())</code>
            
            <h2>Required Features</h2>
            <p>Visit <a href="/api/v1/features">/api/v1/features</a> for complete list</p>
        </div>
    </body>
    </html>
    """
    return docs

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AIoT PASSENGER SAFETY API")
    print("="*60)
    print("\n📡 API Endpoints:")
    print("   Health Check:  http://localhost:5000/health")
    print("   Prediction:    http://localhost:5000/api/v1/predict")
    print("   Documentation: http://localhost:5000/docs")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=True)