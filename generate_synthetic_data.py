"""
Generate Synthetic Test Data for AIoT Safety System
Creates realistic trip data with known risk levels

Run this to create: test_trips.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)  # For reproducibility

# ============================================================================
# GENERATE SYNTHETIC TRIP DATA
# ============================================================================

def generate_safe_trips(n=20):
    """Generate normal, safe trips"""
    trips = []
    for i in range(n):
        trip = {
            'trip_id': f'safe_{i+1}',
            'avg_speed': np.random.uniform(40, 70),  # Normal city speed
            'max_speed': np.random.uniform(60, 90),  # Within limits
            'min_speed': np.random.uniform(0, 20),
            'speed_std': np.random.uniform(8, 18),   # Low variability
            'speed_range': np.random.uniform(40, 70),
            'speed_changes': np.random.uniform(100, 300),
            
            'stop_count': np.random.randint(2, 8),   # Normal stops
            'stop_ratio': np.random.uniform(0.05, 0.25),  # Low stop time
            
            'heading_variance': np.random.uniform(20, 50),  # Normal turns
            'max_deviation': np.random.uniform(0.1, 0.5),
            'avg_route_deviation': np.random.uniform(0.05, 0.3),
            
            'total_distance': np.random.uniform(5, 25),  # km
            'duration_min': np.random.uniform(15, 45),   # minutes
            
            'hour_of_day': np.random.choice([8, 9, 10, 14, 15, 16, 17]),  # Daytime
            'is_night': 0,
            
            'avg_accel': np.random.uniform(-0.5, 0.5),
            'harsh_brake': np.random.randint(0, 2),
            'harsh_accel': np.random.randint(0, 2),
            
            'expected_risk': 'LOW',
            'expected_reason': 'Normal driving behavior'
        }
        trips.append(trip)
    return trips

def generate_aggressive_driving(n=10):
    """Generate aggressive/dangerous driving trips"""
    trips = []
    for i in range(n):
        trip = {
            'trip_id': f'aggressive_{i+1}',
            'avg_speed': np.random.uniform(90, 130),  # High speed
            'max_speed': np.random.uniform(130, 170),  # WAY over limit
            'min_speed': np.random.uniform(20, 40),
            'speed_std': np.random.uniform(35, 55),   # High variability
            'speed_range': np.random.uniform(100, 140),
            'speed_changes': np.random.uniform(500, 800),
            
            'stop_count': np.random.randint(0, 3),    # Few stops
            'stop_ratio': np.random.uniform(0.01, 0.15),  # Almost no stopping
            
            'heading_variance': np.random.uniform(60, 90),  # Swerving
            'max_deviation': np.random.uniform(0.8, 2.0),
            'avg_route_deviation': np.random.uniform(0.5, 1.2),
            
            'total_distance': np.random.uniform(15, 50),
            'duration_min': np.random.uniform(10, 25),  # Fast trips
            
            'hour_of_day': np.random.choice([14, 15, 16, 17, 22, 23]),
            'is_night': int(np.random.choice([0, 0, 0, 1])),
            
            'avg_accel': np.random.uniform(1.5, 3.5),
            'harsh_brake': np.random.randint(5, 15),  # Lots of hard braking
            'harsh_accel': np.random.randint(8, 20),  # Lots of acceleration
            
            'expected_risk': 'HIGH',
            'expected_reason': 'Excessive speed + erratic driving'
        }
        trips.append(trip)
    return trips

def generate_route_hijacking(n=8):
    """Generate suspicious route deviation patterns"""
    trips = []
    for i in range(n):
        trip = {
            'trip_id': f'hijack_{i+1}',
            'avg_speed': np.random.uniform(50, 75),   # Normal speed
            'max_speed': np.random.uniform(70, 100),
            'min_speed': np.random.uniform(0, 10),
            'speed_std': np.random.uniform(18, 35),   # Variable
            'speed_range': np.random.uniform(60, 95),
            'speed_changes': np.random.uniform(300, 600),
            
            'stop_count': np.random.randint(8, 20),   # LOTS of stops
            'stop_ratio': np.random.uniform(0.35, 0.65),  # High stop time
            
            'heading_variance': np.random.uniform(95, 150),  # Crazy route changes
            'max_deviation': np.random.uniform(2.0, 5.0),  # Way off route
            'avg_route_deviation': np.random.uniform(1.5, 3.5),
            
            'total_distance': np.random.uniform(8, 30),
            'duration_min': np.random.uniform(30, 90),  # Long, suspicious
            
            'hour_of_day': np.random.choice([22, 23, 0, 1, 2, 3]),  # Late night
            'is_night': 1,
            
            'avg_accel': np.random.uniform(-1.0, 1.0),
            'harsh_brake': np.random.randint(3, 8),
            'harsh_accel': np.random.randint(2, 6),
            
            'expected_risk': 'HIGH',
            'expected_reason': 'Major route deviation + unusual stops + night time'
        }
        trips.append(trip)
    return trips

def generate_night_risk(n=7):
    """Generate night-time suspicious behavior"""
    trips = []
    for i in range(n):
        trip = {
            'trip_id': f'night_risk_{i+1}',
            'avg_speed': np.random.uniform(45, 70),
            'max_speed': np.random.uniform(65, 95),
            'min_speed': np.random.uniform(0, 15),
            'speed_std': np.random.uniform(15, 28),
            'speed_range': np.random.uniform(50, 80),
            'speed_changes': np.random.uniform(200, 450),
            
            'stop_count': np.random.randint(6, 15),   # Many stops at night
            'stop_ratio': np.random.uniform(0.25, 0.50),
            
            'heading_variance': np.random.uniform(50, 85),
            'max_deviation': np.random.uniform(0.6, 1.5),
            'avg_route_deviation': np.random.uniform(0.4, 1.0),
            
            'total_distance': np.random.uniform(10, 30),
            'duration_min': np.random.uniform(25, 60),
            
            'hour_of_day': np.random.choice([22, 23, 0, 1, 2, 3, 4]),
            'is_night': 1,
            
            'avg_accel': np.random.uniform(-0.5, 1.5),
            'harsh_brake': np.random.randint(2, 7),
            'harsh_accel': np.random.randint(1, 5),
            
            'expected_risk': 'MEDIUM',
            'expected_reason': 'Night time + unusual stop pattern'
        }
        trips.append(trip)
    return trips

def generate_borderline_cases(n=5):
    """Generate edge cases that should be MEDIUM risk"""
    trips = []
    for i in range(n):
        trip = {
            'trip_id': f'borderline_{i+1}',
            'avg_speed': np.random.uniform(75, 95),   # Slightly high
            'max_speed': np.random.uniform(100, 125),  # Near limit
            'min_speed': np.random.uniform(10, 30),
            'speed_std': np.random.uniform(22, 32),
            'speed_range': np.random.uniform(70, 100),
            'speed_changes': np.random.uniform(350, 550),
            
            'stop_count': np.random.randint(4, 10),
            'stop_ratio': np.random.uniform(0.18, 0.35),
            
            'heading_variance': np.random.uniform(55, 75),
            'max_deviation': np.random.uniform(0.5, 1.2),
            'avg_route_deviation': np.random.uniform(0.35, 0.8),
            
            'total_distance': np.random.uniform(12, 35),
            'duration_min': np.random.uniform(20, 50),
            
            'hour_of_day': np.random.choice([10, 11, 18, 19, 20]),
            'is_night': 0,
            
            'avg_accel': np.random.uniform(0.5, 2.0),
            'harsh_brake': np.random.randint(2, 5),
            'harsh_accel': np.random.randint(3, 7),
            
            'expected_risk': 'MEDIUM',
            'expected_reason': 'Moderate speed + some unusual behavior'
        }
        trips.append(trip)
    return trips

# ============================================================================
# GENERATE COMPLETE DATASET
# ============================================================================

print("Generating synthetic test data...")

all_trips = []
all_trips.extend(generate_safe_trips(20))
all_trips.extend(generate_aggressive_driving(10))
all_trips.extend(generate_route_hijacking(8))
all_trips.extend(generate_night_risk(7))
all_trips.extend(generate_borderline_cases(5))

df = pd.DataFrame(all_trips)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('test_trips.csv', index=False)

print(f"✅ Generated {len(df)} synthetic trips")
print(f"   Saved to: test_trips.csv")

# ============================================================================
# GENERATE EXPECTED RESULTS SUMMARY
# ============================================================================

summary = df.groupby('expected_risk').agg({
    'trip_id': 'count',
    'avg_speed': 'mean',
    'max_speed': 'mean',
    'stop_ratio': 'mean',
    'heading_variance': 'mean'
}).round(2)

summary.columns = ['Count', 'Avg Speed', 'Max Speed', 'Stop Ratio', 'Heading Var']

print("\n" + "="*70)
print("EXPECTED DISTRIBUTION")
print("="*70)
print(summary)

print("\n" + "="*70)
print("SAMPLE TRIPS")
print("="*70)
print(df[['trip_id', 'avg_speed', 'max_speed', 'stop_ratio', 'expected_risk', 'expected_reason']].head(10))

# ============================================================================
# GENERATE DIAGNOSTIC SCRIPT
# ============================================================================

diagnostic_code = """
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

print(f"\\nAvailable features: {len(feature_cols)}/{len(feature_names)}")
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
print("\\n" + "="*70)
print("MODEL PERFORMANCE ANALYSIS")
print("="*70)

accuracy = (df['correct'].sum() / len(df) * 100)
print(f"\\nOverall Accuracy: {accuracy:.1f}%")

# Confusion matrix
print("\\nConfusion Matrix:")
confusion = pd.crosstab(
    df['expected_risk'], 
    df['predicted_risk'], 
    rownames=['Expected'], 
    colnames=['Predicted']
)
print(confusion)

# Per-category accuracy
print("\\nPer-Category Performance:")
for risk_level in ['LOW', 'MEDIUM', 'HIGH']:
    subset = df[df['expected_risk'] == risk_level]
    if len(subset) > 0:
        cat_accuracy = (subset['correct'].sum() / len(subset) * 100)
        print(f"  {risk_level:<8} {cat_accuracy:>5.1f}% ({subset['correct'].sum()}/{len(subset)})")

# Score distribution
print("\\nAnomaly Score Statistics by Expected Risk:")
print(df.groupby('expected_risk')['anomaly_score'].describe().round(4))

# Misclassifications
print("\\n" + "="*70)
print("MISCLASSIFIED TRIPS (First 10)")
print("="*70)
misclassified = df[~df['correct']].copy()
if len(misclassified) > 0:
    print(misclassified[['trip_id', 'expected_risk', 'predicted_risk', 
                         'anomaly_score', 'expected_reason']].head(10))
    print(f"\\nTotal misclassifications: {len(misclassified)}")
else:
    print("No misclassifications! Perfect performance! 🎉")

# Save detailed results
df.to_csv('test_results.csv', index=False)
print("\\n✅ Detailed results saved to: test_results.csv")

# Diagnostic insights
print("\\n" + "="*70)
print("DIAGNOSTIC INSIGHTS")
print("="*70)

# Check if model is too conservative
high_expected = df[df['expected_risk'] == 'HIGH']
high_detected = (high_expected['predicted_risk'] == 'HIGH').sum()
print(f"\\nHIGH Risk Detection Rate: {high_detected}/{len(high_expected)} ({high_detected/len(high_expected)*100:.1f}%)")

if high_detected / len(high_expected) < 0.5:
    print("⚠️  Model is UNDER-DETECTING high risk trips!")
    print("   Consider:")
    print("   - Decreasing contamination parameter (currently 0.05)")
    print("   - Adjusting risk thresholds (currently -0.05)")
    print("   - Checking feature scaling")

# Check if model is too aggressive
low_expected = df[df['expected_risk'] == 'LOW']
low_detected = (low_expected['predicted_risk'] == 'LOW').sum()
print(f"\\nLOW Risk Detection Rate: {low_detected}/{len(low_expected)} ({low_detected/len(low_expected)*100:.1f}%)")

if low_detected / len(low_expected) < 0.5:
    print("⚠️  Model is OVER-DETECTING risk (too many false alarms)!")
    print("   Consider:")
    print("   - Increasing contamination parameter")
    print("   - Relaxing risk thresholds")
    print("   - Retraining with more normal trip data")

# Feature importance check
print("\\n" + "="*70)
print("FEATURE STATISTICS")
print("="*70)
print("\\nHigh Risk Trips - Key Features:")
print(df[df['expected_risk'] == 'HIGH'][['avg_speed', 'max_speed', 'stop_ratio', 
                                          'heading_variance']].describe().round(2))

print("\\nLow Risk Trips - Key Features:")
print(df[df['expected_risk'] == 'LOW'][['avg_speed', 'max_speed', 'stop_ratio', 
                                         'heading_variance']].describe().round(2))
"""

with open('diagnose_model.py', 'w') as f:
    f.write(diagnostic_code)

print("\n✅ Created diagnostic script: diagnose_model.py")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("""
1. Run the diagnostic:
   python diagnose_model.py

2. This will show you:
   - Overall accuracy
   - Per-category performance
   - Confusion matrix
   - Misclassified trips
   - Specific recommendations

3. Common issues and fixes:
   
   Issue: Model predicts everything as LOW risk
   Fix: Contamination parameter too high (decrease from 0.05 to 0.10)
        Or risk thresholds too strict
   
   Issue: Model predicts everything as HIGH risk
   Fix: Contamination parameter too low
        Or feature scaling issue
   
   Issue: Poor performance on edge cases (MEDIUM risk)
   Fix: Add more training data
        Or adjust threshold boundaries (-0.05 and 0.05)

4. Upload test_trips.csv to your Streamlit app (Batch Analysis tab)
""")

print("\n📊 Files created:")
print("   - test_trips.csv (50 synthetic trips)")
print("   - diagnose_model.py (performance diagnostic)")
print("\n✅ Ready to test!\n")