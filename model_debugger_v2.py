"""
AIoT Hybrid V2 Model Debugger - COMPREHENSIVE VERSION
Includes Cloud Loading, Feature Analysis, and Diagnostic Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import gdown

# Page config
st.set_page_config(page_title="Hybrid V2 Diagnostic Tool", page_icon="🔧", layout="wide")

st.title("🔧 AIoT Hybrid V2 Comprehensive Debugger")
st.markdown("Diagnose, tune, and analyze the XGBoost + Rule-Based Safety Ensemble")

# ============================================================================
# CLOUD MODEL LOADING
# ============================================================================

GDRIVE_FILE_ID = '1svIVO8BkrPSpHezqmhEXBcQjpGug3nCD'

def download_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with st.spinner(f"☁️ Downloading model from Google Drive..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)

@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'models', 'ghana_safety_ensemble_v2.pkl')
    try:
        if not os.path.exists(model_path):
            download_from_gdrive(GDRIVE_FILE_ID, model_path)
            
        package = joblib.load(model_path)
        return package
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

package = load_model()

if package is None:
    st.error("Model could not be loaded. Please check your internet connection and GDrive File ID.")
    st.stop()

model = package['models']['xgb'] 
scaler = package['scaler']
feature_names = package['feature_names']

# ============================================================================
# SIDEBAR - TUNING & WEIGHTS
# ============================================================================

st.sidebar.header("🛠️ Strategy Tuning")
ml_weight = st.sidebar.slider("AI Influence (ML Prob)", 0.0, 1.0, 0.60, 0.05)
rule_weight = 1.0 - ml_weight

st.sidebar.markdown("---")
st.sidebar.subheader("Safety Thresholds")
thresh_high = st.sidebar.slider("HIGH Risk Threshold", 0.0, 1.0, 0.60, 0.05)
thresh_med = st.sidebar.slider("MEDIUM Risk Threshold", 0.0, 1.0, 0.30, 0.05)

# ============================================================================
# DATA LOADING & HYBRID LOGIC
# ============================================================================

uploaded_file = st.file_uploader("Upload test_trips.csv (must have 'expected_risk' column)", type=['csv'])

if uploaded_file is None:
    st.info("👆 Please upload your ground-truth CSV to begin diagnosis.")
    st.stop()

df = pd.read_csv(uploaded_file)

def calculate_hybrid(row):
    # 1. ML Prob
    input_data = pd.DataFrame([row[feature_names]])
    for col in feature_names:
        if col not in input_data.columns: input_data[col] = 0
        
    X_scaled = scaler.transform(input_data[feature_names])
    ml_prob = model.predict_proba(X_scaled)[0][1]
    
    # 2. Rule Scoring
    rule_score = 0
    triggered = []
    if row.get('max_speed', 0) > 120:
        rule_score += 0.5
        triggered.append("SPEED")
    if row.get('is_night', 0) == 1 and row.get('stop_ratio', 0) > 0.3:
        rule_score += 0.4
        triggered.append("NIGHT_STOP")
    if row.get('speed_std', 0) > row.get('avg_speed', 0) * 0.5:
        rule_score += 0.2
        triggered.append("ERRATIC")

    final_score = min(1.0, (ml_prob * ml_weight) + (rule_score))
    risk = "HIGH" if final_score >= thresh_high else "MEDIUM" if final_score >= thresh_med else "LOW"
    
    return pd.Series([ml_prob, rule_score, final_score, risk, ", ".join(triggered)])

with st.spinner("Processing Hybrid logic..."):
    df[['ml_prob', 'rule_score', 'hybrid_score', 'pred_risk', 'rules_triggered']] = df.apply(calculate_hybrid, axis=1)
    df['is_correct'] = df['pred_risk'] == df['expected_risk']

# ============================================================================
# PERFORMANCE METRICS & VISUALS
# ============================================================================

st.markdown("## 📊 Performance Metrics")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total Accuracy", f"{df['is_correct'].mean() * 100:.1f}%")
with m2:
    high_rec = df[df['expected_risk']=='HIGH']['is_correct'].mean() * 100
    st.metric("HIGH Detection (Recall)", f"{high_rec:.1f}%")
with m3:
    st.metric("Avg Hybrid Score", f"{df['hybrid_score'].mean():.3f}")
with m4:
    fps = df[(df['expected_risk']=='LOW') & (df['pred_risk'] != 'LOW')].shape[0]
    st.metric("False Positives", fps)

tab1, tab2, tab3 = st.tabs(["🎯 Confusion & Error", "📈 Feature Analysis", "🔍 Diagnostic Insights"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        conf = pd.crosstab(df['expected_risk'], df['pred_risk'])
        st.plotly_chart(px.imshow(conf, text_auto=True, title="Confusion Matrix"), use_container_width=True)
    with c2:
        fig_box = px.box(df, x='expected_risk', y='hybrid_score', color='expected_risk', 
                         title="Score Spread per Risk Level", color_discrete_map={'HIGH':'red','MEDIUM':'orange','LOW':'green'})
        fig_box.add_hline(y=thresh_high, line_dash="dash", line_color="red")
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    feat_to_plot = st.selectbox("Select Feature to Analyze", feature_names)
    st.plotly_chart(px.violin(df, x='expected_risk', y=feat_to_plot, color='expected_risk', box=True), use_container_width=True)

with tab3:
    high_median = df[df['expected_risk']=='HIGH']['hybrid_score'].median()
    low_median = df[df['expected_risk']=='LOW']['hybrid_score'].median()
    if abs(high_median - low_median) < 0.2:
        st.error(f"⚠️ Warning: Low Separability ({abs(high_median - low_median):.2f}). Adjust weights.")
    else:
        st.success("✅ Good Separability between risk classes.")

st.markdown("---")
st.subheader("📄 Raw Diagnostic Data")
st.dataframe(df[['expected_risk', 'pred_risk', 'hybrid_score', 'ml_prob', 'rule_score', 'rules_triggered', 'is_correct'] + feature_names])

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Diagnostic Results", csv, "diagnostic_results.csv", "text/csv")