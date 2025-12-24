"""
Interactive Model Debugger and Tuning Tool
Helps you understand and fix model performance issues

Run with: streamlit run model_debugger.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Model Debugger", page_icon="🔧", layout="wide")

st.title("🔧 AIoT Model Debugger & Tuning Tool")
st.markdown("Diagnose and fix model performance issues")

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    try:
        package = joblib.load('models/aiot_detector_v1.pkl')
        return package
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_package = load_model()

if model_package is None:
    st.stop()

model = model_package['iso_forest']
scaler = model_package['scaler']
feature_names = model_package['feature_names']

st.success(f"✅ Model loaded with {len(feature_names)} features")

# ============================================================================
# LOAD TEST DATA
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Load Test Data")

uploaded_file = st.file_uploader("Upload test_trips.csv", type=['csv'])

if uploaded_file is None:
    st.info("👆 Upload test_trips.csv to begin diagnosis")
    st.markdown("""
    **Don't have test data?** Run this first:
    ```python
    python synthetic_test_data.py  # Creates test_trips.csv
    ```
    """)
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"✅ Loaded {len(df)} test trips")

# ============================================================================
# ADJUSTABLE PARAMETERS
# ============================================================================

st.markdown("---")
st.markdown("## ⚙️ Adjust Detection Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    threshold_high = st.slider(
        "HIGH risk threshold (lower = more sensitive)",
        -0.5, 0.0, -0.05, 0.01,
        help="Scores below this are HIGH risk"
    )

with col2:
    threshold_medium = st.slider(
        "MEDIUM risk threshold",
        -0.1, 0.2, 0.05, 0.01,
        help="Scores between HIGH and this are MEDIUM risk"
    )

with col3:
    contamination_display = st.number_input(
        "Contamination parameter",
        0.01, 0.30, 0.05, 0.01,
        help="Expected % of anomalies in training data"
    )

st.info(f"**Current thresholds:** HIGH < {threshold_high:.2f}, MEDIUM < {threshold_medium:.2f}, LOW >= {threshold_medium:.2f}")

# ============================================================================
# RUN PREDICTIONS
# ============================================================================

st.markdown("---")
st.markdown("## 🎯 Model Predictions")

# Ensure all features present
for col in feature_names:
    if col not in df.columns:
        df[col] = 0

# Get predictions
X = df[feature_names]
X_scaled = scaler.transform(X)

df['anomaly_score'] = model.decision_function(X_scaled)
df['is_anomaly'] = model.predict(X_scaled) == -1

# Apply custom thresholds
def score_to_risk(score):
    if score < threshold_high:
        return 'HIGH'
    elif score < threshold_medium:
        return 'MEDIUM'
    else:
        return 'LOW'

df['predicted_risk'] = df['anomaly_score'].apply(score_to_risk)
df['correct'] = df['predicted_risk'] == df['expected_risk']

# Calculate metrics
accuracy = (df['correct'].sum() / len(df) * 100)

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Overall Accuracy", f"{accuracy:.1f}%")

with col2:
    correct_high = df[(df['expected_risk'] == 'HIGH') & df['correct']].shape[0]
    total_high = df[df['expected_risk'] == 'HIGH'].shape[0]
    high_acc = (correct_high / total_high * 100) if total_high > 0 else 0
    st.metric("HIGH Detection", f"{high_acc:.1f}%", 
              f"{correct_high}/{total_high}",
              delta_color="normal" if high_acc >= 70 else "inverse")

with col3:
    correct_low = df[(df['expected_risk'] == 'LOW') & df['correct']].shape[0]
    total_low = df[df['expected_risk'] == 'LOW'].shape[0]
    low_acc = (correct_low / total_low * 100) if total_low > 0 else 0
    st.metric("LOW Detection", f"{low_acc:.1f}%",
              f"{correct_low}/{total_low}",
              delta_color="normal" if low_acc >= 70 else "inverse")

with col4:
    misclassified = len(df[~df['correct']])
    st.metric("Misclassified", misclassified)

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Confusion Matrix")

col1, col2 = st.columns([1, 1])

with col1:
    # Confusion matrix heatmap
    confusion = pd.crosstab(
        df['expected_risk'],
        df['predicted_risk'],
        rownames=['Expected'],
        colnames=['Predicted']
    )
    
    fig = px.imshow(
        confusion.values,
        labels=dict(x="Predicted", y="Expected", color="Count"),
        x=confusion.columns,
        y=confusion.index,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Score distribution by expected risk
    fig = px.box(
        df,
        x='expected_risk',
        y='anomaly_score',
        color='expected_risk',
        color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'},
        title="Anomaly Score Distribution by Expected Risk"
    )
    
    # Add threshold lines
    fig.add_hline(y=threshold_high, line_dash="dash", line_color="red",
                  annotation_text="HIGH threshold")
    fig.add_hline(y=threshold_medium, line_dash="dash", line_color="orange",
                  annotation_text="MEDIUM threshold")
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DIAGNOSTIC INSIGHTS
# ============================================================================

st.markdown("---")
st.markdown("## 🔍 Diagnostic Insights")

# Check for common issues
issues = []
recommendations = []

# Issue 1: Poor HIGH risk detection
if high_acc < 50:
    issues.append("⚠️ **CRITICAL**: Model is missing most HIGH risk trips!")
    recommendations.append(f"""
    **Fix for LOW HIGH-risk detection ({high_acc:.1f}%):**
    - Move HIGH threshold UP (currently {threshold_high:.2f} → try -0.10 or -0.15)
    - Model is being too conservative
    - Check if HIGH risk trips have extreme feature values
    """)

# Issue 2: Too many false positives
if low_acc < 50:
    issues.append("⚠️ **WARNING**: Too many false alarms on safe trips!")
    recommendations.append(f"""
    **Fix for LOW safe-trip accuracy ({low_acc:.1f}%):**
    - Move HIGH threshold DOWN (currently {threshold_high:.2f} → try -0.03)
    - Model is being too aggressive
    - Consider retraining with contamination={contamination_display + 0.05:.2f}
    """)

# Issue 3: Score overlap
high_trips = df[df['expected_risk'] == 'HIGH']
low_trips = df[df['expected_risk'] == 'LOW']

if len(high_trips) > 0 and len(low_trips) > 0:
    high_median = high_trips['anomaly_score'].median()
    low_median = low_trips['anomaly_score'].median()
    
    if abs(high_median - low_median) < 0.1:
        issues.append("⚠️ **WARNING**: Anomaly scores overlap significantly!")
        recommendations.append(f"""
        **Fix for overlapping scores:**
        - HIGH median: {high_median:.4f}, LOW median: {low_median:.4f}
        - Model cannot clearly distinguish between risk levels
        - **Retrain model** with more diverse training data
        - Check if features are properly scaled
        """)

# Display issues
if issues:
    st.error("### Issues Detected:")
    for issue in issues:
        st.markdown(issue)
    
    st.warning("### Recommendations:")
    for rec in recommendations:
        st.markdown(rec)
else:
    st.success("### ✅ No major issues detected! Model performance is good.")

# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Feature Analysis")

# Key features comparison
key_features = ['avg_speed', 'max_speed', 'stop_ratio', 'heading_variance']
available_key_features = [f for f in key_features if f in df.columns]

if available_key_features:
    st.markdown("### Feature Values by Expected Risk Level")
    
    fig = go.Figure()
    
    for risk_level in ['LOW', 'MEDIUM', 'HIGH']:
        subset = df[df['expected_risk'] == risk_level]
        for feature in available_key_features[:4]:  # Show top 4
            values = subset[feature].values
            fig.add_trace(go.Box(
                y=values,
                name=f"{risk_level} - {feature}",
                boxmean='sd'
            ))
    
    fig.update_layout(
        title="Feature Distributions by Risk Level",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MISCLASSIFIED TRIPS
# ============================================================================

st.markdown("---")
st.markdown("## ❌ Misclassified Trips")

misclassified = df[~df['correct']].copy()

if len(misclassified) > 0:
    st.warning(f"Found {len(misclassified)} misclassified trips")
    
    # Group by type of error
    tab1, tab2, tab3 = st.tabs(["All Misclassifications", "False Negatives", "False Positives"])
    
    with tab1:
        st.dataframe(
            misclassified[['trip_id', 'expected_risk', 'predicted_risk', 
                          'anomaly_score', 'avg_speed', 'max_speed', 
                          'stop_ratio', 'heading_variance', 'expected_reason']],
            use_container_width=True
        )
    
    with tab2:
        # HIGH expected but not detected
        false_neg = misclassified[
            (misclassified['expected_risk'] == 'HIGH') & 
            (misclassified['predicted_risk'] != 'HIGH')
        ]
        st.markdown(f"**{len(false_neg)} dangerous trips missed**")
        if len(false_neg) > 0:
            st.dataframe(
                false_neg[['trip_id', 'anomaly_score', 'expected_reason', 
                          'avg_speed', 'max_speed', 'stop_ratio']],
                use_container_width=True
            )
    
    with tab3:
        # LOW expected but flagged as HIGH
        false_pos = misclassified[
            (misclassified['expected_risk'] == 'LOW') & 
            (misclassified['predicted_risk'] == 'HIGH')
        ]
        st.markdown(f"**{len(false_pos)} false alarms on safe trips**")
        if len(false_pos) > 0:
            st.dataframe(
                false_pos[['trip_id', 'anomaly_score', 'avg_speed', 
                          'max_speed', 'stop_ratio']],
                use_container_width=True
            )
else:
    st.success("🎉 Perfect performance! No misclassifications!")

# ============================================================================
# THRESHOLD TUNING HELPER
# ============================================================================

st.markdown("---")
st.markdown("## 🎛️ Threshold Tuning Assistant")

st.markdown("""
Use this tool to find optimal thresholds for your use case.

**Trade-offs:**
- **Lower HIGH threshold** (more negative) = Catch more dangerous trips BUT more false alarms
- **Higher HIGH threshold** (less negative) = Fewer false alarms BUT miss some dangerous trips
""")

# Show score ranges
col1, col2, col3 = st.columns(3)

with col1:
    if len(high_trips) > 0:
        st.metric("HIGH Risk Score Range",
                 f"{high_trips['anomaly_score'].min():.3f} to {high_trips['anomaly_score'].max():.3f}")

with col2:
    medium_trips = df[df['expected_risk'] == 'MEDIUM']
    if len(medium_trips) > 0:
        st.metric("MEDIUM Risk Score Range",
                 f"{medium_trips['anomaly_score'].min():.3f} to {medium_trips['anomaly_score'].max():.3f}")

with col3:
    if len(low_trips) > 0:
        st.metric("LOW Risk Score Range",
                 f"{low_trips['anomaly_score'].min():.3f} to {low_trips['anomaly_score'].max():.3f}")

# Suggest optimal thresholds
if len(high_trips) > 0 and len(low_trips) > 0:
    # Find threshold that maximizes both
    high_75th = high_trips['anomaly_score'].quantile(0.75)
    low_25th = low_trips['anomaly_score'].quantile(0.25)
    
    suggested_threshold = (high_75th + low_25th) / 2
    
    st.info(f"""
    **Suggested HIGH threshold:** {suggested_threshold:.3f}
    
    This balances between:
    - 75% of HIGH risk trips have scores < {high_75th:.3f}
    - 25% of LOW risk trips have scores < {low_25th:.3f}
    """)

# ============================================================================
# EXPORT RESULTS
# ============================================================================

st.markdown("---")
st.markdown("## 💾 Export Results")

if st.button("📥 Generate Full Diagnostic Report"):
    # Save detailed results
    df.to_csv('detailed_results.csv', index=False)
    
    # Generate report
    report = f"""
# AIoT Model Diagnostic Report
Generated: {pd.Timestamp.now()}

## Performance Summary
- Overall Accuracy: {accuracy:.2f}%
- HIGH Risk Detection: {high_acc:.2f}%
- LOW Risk Detection: {low_acc:.2f}%
- Total Misclassifications: {len(misclassified)}

## Current Thresholds
- HIGH risk threshold: {threshold_high:.3f}
- MEDIUM risk threshold: {threshold_medium:.3f}

## Issues Detected
{chr(10).join(issues) if issues else 'No issues detected'}

## Recommendations
{chr(10).join(recommendations) if recommendations else 'Model performing well'}

## Score Statistics
{df.groupby('expected_risk')['anomaly_score'].describe().to_string()}

## Confusion Matrix
{confusion.to_string()}
"""
    
    st.download_button(
        "📄 Download Report (TXT)",
        report,
        "diagnostic_report.txt",
        "text/plain"
    )
    
    st.download_button(
        "📊 Download Detailed Results (CSV)",
        df.to_csv(index=False),
        "detailed_results.csv",
        "text/csv"
    )
    
    st.success("✅ Report generated!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 💡 Quick Fixes Guide

| Problem | Solution |
|---------|----------|
| Missing HIGH risk trips | Increase HIGH threshold (less negative) |
| Too many false alarms | Decrease HIGH threshold (more negative) |
| Scores overlap | Retrain model with better training data |
| Poor MEDIUM detection | Adjust MEDIUM threshold |
| All predictions same | Check feature scaling / retrain model |

**Need more help?** Check the training notebook for retraining options.
""")