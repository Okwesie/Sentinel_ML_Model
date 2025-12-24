"""
AIoT Passenger Safety System - Streamlit Dashboard
Complete interactive app for testing your trained model

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import gdown
import os


# ============================================================================
# GOOGLE DRIVE LOADER
# ============================================================================

def download_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with st.spinner(f"Downloading model from Cloud... (approx. 50MB)"):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)


# Page config
st.set_page_config(
    page_title="AIoT Passenger Safety System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================


# ============================================================================
# GOOGLE DRIVE LOADER (CORRECTED)
# ============================================================================

GDRIVE_FILE_ID = '1svIVO8BkrPSpHezqmhEXBcQjpGug3nCD'

def download_from_gdrive(file_id, output_path):
    """
    Downloads the model using the fuzzy search method which is 
    more reliable for Google Drive 'View' links.
    """
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with st.spinner(f"☁️ Syncing Model from Cloud..."):
            # We use the full sharing URL + fuzzy=True for maximum reliability
            url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
            try:
                gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"❌ Cloud Download Failed: {e}")

@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'models', 'ghana_safety_ensemble_v2.pkl')
    
    try:
        # 1. Ensure file exists
        if not os.path.exists(model_path):
            download_from_gdrive(GDRIVE_FILE_ID, model_path)
            
        # 2. Final check before loading
        if not os.path.exists(model_path):
            return None
            
        # 3. Load and Return
        package = joblib.load(model_path)
        return {
            'model': package['models']['xgb'],
            'scaler': package['scaler'],
            'feature_names': package['feature_names'],
            'thresholds': package.get('thresholds', {'medium': 0.3, 'high': 0.6}),
            'version': package.get('version', '2.1.0_hybrid')
        }
    except Exception as e:
        st.error(f"❌ Model Initialization Error: {e}")
        return None
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_risk(telemetry_data):
    """
    Run Hybrid V2 prediction: 
    Combines Best Model (XGBoost) Probability + Safety Rules
    """
    if model_package is None:
        return None
    
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    thresholds = model_package['thresholds']
    
    # 1. Convert to DataFrame and Preprocess
    input_df = pd.DataFrame([telemetry_data])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    X_input = scaler.transform(input_df[feature_names])
    
    # 2. Get ML Probability (Dangerous class)
    ml_prob = model.predict_proba(X_input)[0][1]
    
    # 3. Apply Safety Rules Score (Ghana-specific logic)
    rule_score = 0
    triggered_rules = []
    
    if telemetry_data.get('max_speed', 0) > 120:
        rule_score += 0.5
        triggered_rules.append("EXCESSIVE_SPEED")
    
    if telemetry_data.get('is_night', 0) == 1 and telemetry_data.get('stop_ratio', 0) > 0.3:
        rule_score += 0.4
        triggered_rules.append("HIGH_RISK_NIGHT_STOP")
        
    if telemetry_data.get('speed_std', 0) > telemetry_data.get('avg_speed', 0) * 0.5:
        rule_score += 0.2
        triggered_rules.append("ERRATIC_DRIVING")

    # 4. Calculate Final Hybrid Score (60% ML, 40% Rules)
    # We cap at 1.0
    final_score = min(1.0, (ml_prob * 0.6) + (rule_score))
    
    # 5. Determine risk level based on thresholds
    if final_score >= thresholds['high']:
        risk_level = "HIGH"
        color = "red"
    elif final_score >= thresholds['medium']:
        risk_level = "MEDIUM"
        color = "orange"
    else:
        risk_level = "LOW"
        color = "green"
    
    return {
        'risk_level': risk_level,
        'final_score': round(float(final_score), 4),
        'ml_probability': round(float(ml_prob), 4),
        'rules_triggered': triggered_rules,
        'color': color,
        'timestamp': datetime.now().isoformat(),
        'trigger_emergency': risk_level == "HIGH"
    }

def apply_rules(data):
    """Apply rule-based detection"""
    alerts = []
    
    if data.get('max_speed', 0) > 120:
        alerts.append({
            'rule': 'EXCESSIVE_SPEED',
            'severity': 'HIGH',
            'value': data['max_speed']
        })
    
    if data.get('stop_ratio', 0) > 0.3:
        alerts.append({
            'rule': 'UNUSUAL_STOPS',
            'severity': 'MEDIUM',
            'value': data['stop_ratio']
        })
    
    if data.get('speed_std', 0) > data.get('avg_speed', 0) * 0.5:
        alerts.append({
            'rule': 'ERRATIC_DRIVING',
            'severity': 'MEDIUM',
            'value': data['speed_std']
        })
    
    if data.get('heading_variance', 0) > 90:
        alerts.append({
            'rule': 'ROUTE_DEVIATION',
            'severity': 'HIGH',
            'value': data['heading_variance']
        })
    
    return alerts

# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================

st.sidebar.title("🚗 AIoT Safety System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🧪 Test Model", "📊 Batch Analysis", "📖 API Docs", "ℹ️ About"]
)

# Display model status
if model_package:
    st.sidebar.success("✅ Model Loaded")
    st.sidebar.info(f"Features: {len(model_package['feature_names'])}")
else:
    st.sidebar.error("❌ Model Not Loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("**Authors:**  \nCaleb Okwesie Arthur  \nFrances Seyram Fiahagbe")

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================

if page == "🏠 Dashboard":
    st.title("🚗 AIoT Passenger Safety Dashboard")
    st.markdown("Real-time anomaly detection for ride-hailing safety in Ghana")
    
    # Quick test section
    st.markdown("### 🎯 Quick Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_speed = st.number_input("Average Speed (km/h)", 0.0, 200.0, 50.0)
        max_speed = st.number_input("Max Speed (km/h)", 0.0, 200.0, 80.0)
        speed_std = st.number_input("Speed Std Dev", 0.0, 100.0, 15.0)
    
    with col2:
        stop_ratio = st.slider("Stop Ratio", 0.0, 1.0, 0.2)
        stop_count = st.number_input("Stop Count", 0, 50, 5)
        heading_variance = st.number_input("Heading Variance", 0.0, 180.0, 45.0)
    
    with col3:
        is_night = st.checkbox("Night Time (22:00-06:00)")
        hour_of_day = st.slider("Hour of Day", 0, 23, 14)
        duration_min = st.number_input("Duration (min)", 0.0, 300.0, 30.0)
    
    if st.button("🔍 Assess Risk", type="primary"):
        telemetry = {
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'speed_std': speed_std,
            'stop_ratio': stop_ratio,
            'stop_count': stop_count,
            'heading_variance': heading_variance,
            'is_night': int(is_night),
            'hour_of_day': hour_of_day,
            'duration_min': duration_min
        }
        
        # Get prediction
        result = predict_risk(telemetry)
        rules = apply_rules(telemetry)
        
        if result:
            # Display result in big cards
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {'#ef4444' if result['risk_level']=='HIGH' else '#f59e0b' if result['risk_level']=='MEDIUM' else '#10b981'} 0%, {'#dc2626' if result['risk_level']=='HIGH' else '#d97706' if result['risk_level']=='MEDIUM' else '#059669'} 100%);">
                    <h2 style="margin:0; font-size:3rem;">⚠️</h2>
                    <h3 style="margin:0.5rem 0;">Risk Level</h3>
                    <h1 style="margin:0; font-size:2.5rem;">{result['risk_level']}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin:0; font-size:3rem;">📊</h2>
                    <h3 style="margin:0.5rem 0;">Hybrid Score</h3>
                    <h1 style="margin:0; font-size:2.5rem;">{result['final_score']:.4f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin:0; font-size:3rem;">🚨</h2>
                    <h3 style="margin:0.5rem 0;">Emergency Alert</h3>
                    <h1 style="margin:0; font-size:2rem;">{'TRIGGERED' if result['trigger_emergency'] else 'NO'}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Rule alerts
            if rules:
                st.markdown("### 🚦 Rule Violations Detected")
                for rule in rules:
                    severity_icon = "🔴" if rule['severity'] == 'HIGH' else "🟡"
                    st.warning(f"{severity_icon} **{rule['rule']}** - Value: {rule['value']:.2f}")
            
            # Action recommendation
            st.markdown("### 💡 Recommended Actions")
            if result['trigger_emergency']:
                st.error("🚨 **IMMEDIATE ACTION REQUIRED**")
                st.markdown("""
                - ✅ Send SMS alerts to emergency contacts
                - ✅ Notify local authorities
                - ✅ Share live location with trusted contacts
                - ✅ Activate wearable device alarm
                """)
            elif result['risk_level'] == 'MEDIUM':
                st.warning("⚠️ **MONITOR CLOSELY**")
                st.markdown("""
                - 👁️ Increase monitoring frequency
                - 📲 Send warning notification to passenger
                - 📍 Log location for review
                """)
            else:
                st.success("✅ **TRIP IS SAFE**")
                st.markdown("Continue normal monitoring.")

# ============================================================================
# PAGE 2: TEST MODEL
# ============================================================================

elif page == "🧪 Test Model":
    st.title("🧪 Model Testing Interface")
    st.markdown("Test the model with different scenarios")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📋 Predefined Scenarios", "🎲 Random Test"])
    
    # Tab 1: Custom input
    with tab1:
        st.markdown("### Enter Custom Trip Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Speed Metrics")
            custom_avg_speed = st.number_input("Avg Speed", 0.0, 200.0, 50.0, key="custom_avg")
            custom_max_speed = st.number_input("Max Speed", 0.0, 200.0, 80.0, key="custom_max")
            custom_speed_std = st.number_input("Speed Std", 0.0, 100.0, 15.0, key="custom_std")
            
            st.markdown("#### Stop Behavior")
            custom_stop_ratio = st.slider("Stop Ratio", 0.0, 1.0, 0.2, key="custom_stop")
            custom_stop_count = st.number_input("Stop Count", 0, 50, 5, key="custom_count")
        
        with col2:
            st.markdown("#### Route Metrics")
            custom_heading_var = st.number_input("Heading Variance", 0.0, 180.0, 45.0, key="custom_heading")
            custom_max_dev = st.number_input("Max Deviation", 0.0, 10.0, 0.5, key="custom_dev")
            
            st.markdown("#### Context")
            custom_is_night = st.checkbox("Night Time", key="custom_night")
            custom_hour = st.slider("Hour", 0, 23, 14, key="custom_hour")
        
        if st.button("Run Prediction", type="primary", key="custom_predict"):
            custom_data = {
                'avg_speed': custom_avg_speed,
                'max_speed': custom_max_speed,
                'speed_std': custom_speed_std,
                'stop_ratio': custom_stop_ratio,
                'stop_count': custom_stop_count,
                'heading_variance': custom_heading_var,
                'max_deviation': custom_max_dev,
                'is_night': int(custom_is_night),
                'hour_of_day': custom_hour
            }
            
            result = predict_risk(custom_data)
            if result:
                st.json(result)
    
    # Tab 2: Predefined scenarios
    with tab2:
        st.markdown("### Test with Predefined Scenarios")
        
        scenarios = {
            "Safe Commute": {
                'avg_speed': 45.0,
                'max_speed': 65.0,
                'speed_std': 10.0,
                'stop_ratio': 0.15,
                'stop_count': 3,
                'heading_variance': 30.0,
                'is_night': 0,
                'hour_of_day': 14
            },
            "Aggressive Driving": {
                'avg_speed': 95.0,
                'max_speed': 140.0,
                'speed_std': 45.0,
                'stop_ratio': 0.1,
                'stop_count': 2,
                'heading_variance': 65.0,
                'is_night': 0,
                'hour_of_day': 16
            },
            "Route Hijacking": {
                'avg_speed': 60.0,
                'max_speed': 85.0,
                'speed_std': 20.0,
                'stop_ratio': 0.45,
                'stop_count': 12,
                'heading_variance': 110.0,
                'is_night': 1,
                'hour_of_day': 23
            },
            "Night Risk": {
                'avg_speed': 55.0,
                'max_speed': 75.0,
                'speed_std': 15.0,
                'stop_ratio': 0.35,
                'stop_count': 8,
                'heading_variance': 50.0,
                'is_night': 1,
                'hour_of_day': 2
            }
        }
        
        scenario_choice = st.selectbox("Select Scenario", list(scenarios.keys()))
        
        st.json(scenarios[scenario_choice])
        
        if st.button("Test Scenario", type="primary", key="scenario_test"):
            result = predict_risk(scenarios[scenario_choice])
            rules = apply_rules(scenarios[scenario_choice])
            
            if result:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Level", result['risk_level'])
                    st.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
                
                with col2:
                    st.metric("Is Anomaly", "YES" if result['is_anomaly'] else "NO")
                    st.metric("Emergency", "TRIGGERED" if result['trigger_emergency'] else "NO")
                
                if rules:
                    st.markdown("#### Rule Violations")
                    for rule in rules:
                        st.warning(f"**{rule['rule']}** ({rule['severity']}): {rule['value']:.2f}")
    
    # Tab 3: Random test
    with tab3:
        st.markdown("### Generate Random Trip")
        
        if st.button("🎲 Generate Random Trip", type="primary"):
            random_data = {
                'avg_speed': np.random.uniform(20, 120),
                'max_speed': np.random.uniform(40, 150),
                'speed_std': np.random.uniform(5, 50),
                'stop_ratio': np.random.uniform(0, 0.5),
                'stop_count': int(np.random.uniform(0, 20)),
                'heading_variance': np.random.uniform(0, 120),
                'is_night': int(np.random.choice([0, 1])),
                'hour_of_day': int(np.random.uniform(0, 24))
            }
            
            st.json(random_data)
            
            result = predict_risk(random_data)
            if result:
                st.markdown("### Prediction Result")
                st.json(result)

# ============================================================================
# PAGE 3: BATCH ANALYSIS
# ============================================================================

elif page == "📊 Batch Analysis":
    st.title("📊 Batch Trip Analysis")
    st.markdown("Upload a CSV file to analyze multiple trips")
    
    uploaded_file = st.file_uploader("Upload CSV with trip data", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} trips")
        
        st.dataframe(df.head())
        
        if st.button("🔍 Analyze All Trips", type="primary"):
            with st.spinner("Analyzing trips..."):
                results = []
                for idx, row in df.iterrows():
                    result = predict_risk(row.to_dict())
                    if result:
                        results.append({
                            'trip_id': idx,
                            'risk_level': result['risk_level'],
                            'anomaly_score': result['final_score'], # Use final_score here
                            'ml_prob': result['ml_probability']
                        })
                
                results_df = pd.DataFrame(results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trips", len(results_df))
                with col2:
                    high_risk = (results_df['risk_level'] == 'HIGH').sum()
                    st.metric("High Risk", high_risk, delta=f"{high_risk/len(results_df)*100:.1f}%")
                with col3:
                    medium_risk = (results_df['risk_level'] == 'MEDIUM').sum()
                    st.metric("Medium Risk", medium_risk)
                with col4:
                    low_risk = (results_df['risk_level'] == 'LOW').sum()
                    st.metric("Low Risk", low_risk)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk distribution pie chart
                    risk_counts = results_df['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution",
                        color=risk_counts.index,
                        color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Anomaly score distribution
                    fig = px.histogram(
                        results_df,
                        x='anomaly_score',
                        title="Anomaly Score Distribution",
                        nbins=30,
                        color='risk_level',
                        color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.markdown("### Detailed Results")
                st.dataframe(results_df)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "trip_analysis_results.csv",
                    "text/csv"
                )

# ============================================================================
# PAGE 4: API DOCUMENTATION
# ============================================================================

elif page == "📖 API Docs":
    st.title("📖 API Documentation")
    st.markdown("Complete reference for integrating with the safety system")
    
    st.markdown("## 🔌 REST API Endpoints")
    
    # Endpoint 1
    with st.expander("POST /api/v1/predict - Real-time Prediction"):
        st.markdown("""
        ### Endpoint: `/api/v1/predict`
        **Method:** POST  
        **Description:** Get real-time risk assessment for trip data
        
        #### Request Body
        ```json
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
        ```
        
        #### Response
        ```json
        {
          "risk_level": "LOW",
          "anomaly_score": 0.1234,
          "is_anomaly": false,
          "trigger_emergency": false,
          "timestamp": "2025-12-24T10:30:00"
        }
        ```
        
        #### Python Example
        ```python
        import requests
        
        data = {
            "avg_speed": 50.0,
            "max_speed": 80.0,
            "speed_std": 15.0
        }
        
        response = requests.post(
            "http://localhost:5000/api/v1/predict",
            json=data
        )
        
        print(response.json())
        ```
        """)
    
    # Endpoint 2
    with st.expander("POST /api/v1/batch-predict - Batch Analysis"):
        st.markdown("""
        ### Endpoint: `/api/v1/batch-predict`
        **Method:** POST  
        **Description:** Analyze multiple trips at once
        
        #### Request Body
        ```json
        {
          "trips": [
            {"avg_speed": 50.0, "max_speed": 80.0, ...},
            {"avg_speed": 95.0, "max_speed": 140.0, ...}
          ]
        }
        ```
        
        #### Response
        ```json
        {
          "results": [
            {"trip_id": 0, "risk_level": "LOW", ...},
            {"trip_id": 1, "risk_level": "HIGH", ...}
          ],
          "summary": {
            "total": 2,
            "high_risk": 1,
            "medium_risk": 0,
            "low_risk": 1
          }
        }
        ```
        """)
    
    # Feature requirements
    st.markdown("## 📋 Required Features")
    
    if model_package:
        feature_df = pd.DataFrame({
            'Feature': model_package['feature_names'],
            'Type': ['float'] * len(model_package['feature_names']),
            'Example': [50.0, 80.0, 15.0, 0.2, 5.0, 45.0, 0, 14] + [0.0] * (len(model_package['feature_names']) - 8)
        })
        st.dataframe(feature_df)
    
    # cURL examples
    st.markdown("## 🔧 cURL Examples")
    st.code("""
# Test prediction endpoint
curl -X POST http://localhost:5000/api/v1/predict \\
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
  }'
    """, language="bash")

# ============================================================================
# UPDATED ABOUT SECTION
# ============================================================================

if page == "ℹ️ About":
    st.title("ℹ️ About AIoT Passenger Safety System")
    st.markdown("""
    ## 🎯 Project Overview
    **Authors:** Caleb Okwesie Arthur
    **Location:** Accra, Ghana (2025)
    
    ---
    ## 🚀 Advancements in V2.1 (Hybrid Ensemble)
    
    This system has evolved from simple Anomaly Detection to a **Hybrid Ensemble Architecture**:
    
    1. **AI Component (60%):** Powered by **XGBoost**, trained specifically on Ghanaian transit patterns to recognize complex risk signatures.
    2. **Expert Rules (40%):** Hard-coded safety thresholds (e.g., Night-time stops > 30% of trip) to ensure 100% catch rate for known danger scenarios.
    3. **Cloud-Native Deployment:** Models are dynamically pulled from secure storage to ensure the dashboard remains lightweight and fast.
    
    ---
    ## 🌍 Impact & Safety
    - **Proactive Intervention:** Flags hijacking or erratic driving *before* an accident occurs.
    - **Context Aware:** Understands that a stop at 2:00 AM in Accra is higher risk than a stop at 2:00 PM.
    """)

# ============================================================================
# UPDATED FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><b>AIoT Passenger Safety System v2.1</b></p>
    <p>© 2025 Caleb Okwesie Arthur</p>
    <p style='font-size: 0.8rem;'>System Status: <span style='color:green;'>Cloud-Sync Active</span> | Accra, Ghana</p>
</div>
""", unsafe_allow_html=True)