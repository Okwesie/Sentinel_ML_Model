
'''
Test Your Model with Synthetic Data
Diagnoses model performance issues
'''

import pandas as pd
import joblib
import numpy as np

# Load test data
df = pd.read_csv('test_trips.csv')
print(f"Loaded {len(df)} test trips")

# Load model
try:
    package = joblib.load('models/aiot_detector_v1.pkl')
    model = package['iso_forest']
    scaler = package['scaler']
    feature_names = package['feature_names']
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Prepare features
feature_cols = [col for col in feature_names if col in df.columns]
missing_features = [col for col in feature_names if col not in df.columns]

print(f"\nAvailable features: {len(feature_cols)}/{len(feature_names)}")
if missing_features:
    print(f"Missing features: {missing_features}")
    # Fill missing with 0
    for col in missing_features:
        df[col] = 0

# Get predictions
X = df[feature_names]
X_scaled = scaler.transform(X)

df['anomaly_score'] = model.decision_function(X_scaled)
df['is_anomaly'] = model.predict(X_scaled) == -1

# Map to risk levels
def score_to_risk(score):
    if score < -0.05:
        return 'HIGH'
    elif score < 0.05:
        return 'MEDIUM'
    else:
        return 'LOW'

df['predicted_risk'] = df['anomaly_score'].apply(score_to_risk)

# Compare predictions vs expectations
df['correct'] = df['predicted_risk'] == df['expected_risk']

# Results
print("\n" + "="*70)
print("MODEL PERFORMANCE ANALYSIS")
print("="*70)

accuracy = (df['correct'].sum() / len(df) * 100)
print(f"\nOverall Accuracy: {accuracy:.1f}%")

# Confusion matrix
print("\nConfusion Matrix:")
confusion = pd.crosstab(
    df['expected_risk'], 
    df['predicted_risk'], 
    rownames=['Expected'], 
    colnames=['Predicted']
)
print(confusion)

# Per-category accuracy
print("\nPer-Category Performance:")
for risk_level in ['LOW', 'MEDIUM', 'HIGH']:
    subset = df[df['expected_risk'] == risk_level]
    if len(subset) > 0:
        cat_accuracy = (subset['correct'].sum() / len(subset) * 100)
        print(f"  {risk_level:<8} {cat_accuracy:>5.1f}% ({subset['correct'].sum()}/{len(subset)})")

# Score distribution
print("\nAnomaly Score Statistics by Expected Risk:")
print(df.groupby('expected_risk')['anomaly_score'].describe().round(4))

# Misclassifications
print("\n" + "="*70)
print("MISCLASSIFIED TRIPS (First 10)")
print("="*70)
misclassified = df[~df['correct']].copy()
if len(misclassified) > 0:
    print(misclassified[['trip_id', 'expected_risk', 'predicted_risk', 
                         'anomaly_score', 'expected_reason']].head(10))
    print(f"\nTotal misclassifications: {len(misclassified)}")
else:
    print("No misclassifications! Perfect performance! 🎉")

# Save detailed results
df.to_csv('test_results.csv', index=False)
print("\n✅ Detailed results saved to: test_results.csv")

# Diagnostic insights
print("\n" + "="*70)
print("DIAGNOSTIC INSIGHTS")
print("="*70)

# Check if model is too conservative
high_expected = df[df['expected_risk'] == 'HIGH']
high_detected = (high_expected['predicted_risk'] == 'HIGH').sum()
print(f"\nHIGH Risk Detection Rate: {high_detected}/{len(high_expected)} ({high_detected/len(high_expected)*100:.1f}%)")

if high_detected / len(high_expected) < 0.5:
    print("⚠️  Model is UNDER-DETECTING high risk trips!")
    print("   Consider:")
    print("   - Decreasing contamination parameter (currently 0.05)")
    print("   - Adjusting risk thresholds (currently -0.05)")
    print("   - Checking feature scaling")

# Check if model is too aggressive
low_expected = df[df['expected_risk'] == 'LOW']
low_detected = (low_expected['predicted_risk'] == 'LOW').sum()
print(f"\nLOW Risk Detection Rate: {low_detected}/{len(low_expected)} ({low_detected/len(low_expected)*100:.1f}%)")

if low_detected / len(low_expected) < 0.5:
    print("⚠️  Model is OVER-DETECTING risk (too many false alarms)!")
    print("   Consider:")
    print("   - Increasing contamination parameter")
    print("   - Relaxing risk thresholds")
    print("   - Retraining with more normal trip data")

# Feature importance check
print("\n" + "="*70)
print("FEATURE STATISTICS")
print("="*70)
print("\nHigh Risk Trips - Key Features:")
print(df[df['expected_risk'] == 'HIGH'][['avg_speed', 'max_speed', 'stop_ratio', 
                                          'heading_variance']].describe().round(2))

print("\nLow Risk Trips - Key Features:")
print(df[df['expected_risk'] == 'LOW'][['avg_speed', 'max_speed', 'stop_ratio', 
                                         'heading_variance']].describe().round(2))
