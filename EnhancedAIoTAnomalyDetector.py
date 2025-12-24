"""
Enhanced AIoT Anomaly Detection System for Passenger Safety
Combining best practices from both approaches with production-ready features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedAIoTAnomalyDetector:
    """
    Complete anomaly detection system with:
    - Multi-source feature engineering
    - Hybrid detection (ML + Rules + Clustering)
    - Real-time inference optimization
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = None
        self.dbscan = None
        self.scaler = StandardScaler()
        self.le_dict = {}
        self.feature_names = []
        self.cluster_profiles = {}
        
    def engineer_trip_features(self, df):
        """
        Trip-level aggregation features from GPS/telemetry data
        Extracts behavioral patterns from time-series data
        """
        print("🔧 Engineering trip-level features...")
        
        trip_features = pd.DataFrame()
        
        for trip_id in df['trip_id'].unique():
            trip_data = df[df['trip_id'] == trip_id].copy()
            
            if 'timestamp' in trip_data.columns:
                trip_data = trip_data.sort_values('timestamp')
                trip_data['timestamp'] = pd.to_datetime(trip_data['timestamp'])
            
            idx = trip_id
            
            # === SPEED-BASED FEATURES ===
            trip_features.loc[idx, 'avg_speed'] = trip_data['speed'].mean()
            trip_features.loc[idx, 'max_speed'] = trip_data['speed'].max()
            trip_features.loc[idx, 'min_speed'] = trip_data['speed'].min()
            trip_features.loc[idx, 'speed_variability'] = trip_data['speed'].std()
            trip_features.loc[idx, 'speed_changes'] = trip_data['speed'].diff().abs().sum()
            
            # Acceleration patterns
            if 'acceleration' in trip_data.columns:
                trip_features.loc[idx, 'avg_acceleration'] = trip_data['acceleration'].mean()
                trip_features.loc[idx, 'harsh_braking_count'] = (trip_data['acceleration'] < -3).sum()
                trip_features.loc[idx, 'harsh_accel_count'] = (trip_data['acceleration'] > 3).sum()
            
            # === STOP DETECTION ===
            stops = (trip_data['speed'] < 0.5).sum()
            trip_features.loc[idx, 'stop_count'] = stops
            trip_features.loc[idx, 'stop_ratio'] = stops / len(trip_data) if len(trip_data) > 0 else 0
            
            # Prolonged stops (potential danger indicator)
            if 'stop_events' in trip_data.columns:
                trip_features.loc[idx, 'prolonged_stops'] = trip_data['stop_events'].sum()
            
            # === ROUTE DEVIATION FEATURES ===
            if 'latitude' in trip_data.columns and 'longitude' in trip_data.columns:
                trip_features.loc[idx, 'lat_std'] = trip_data['latitude'].std()
                trip_features.loc[idx, 'lon_std'] = trip_data['longitude'].std()
            
            if 'bearing' in trip_data.columns:
                trip_features.loc[idx, 'bearing_variance'] = trip_data['bearing'].std()
            elif 'heading' in trip_data.columns:
                trip_features.loc[idx, 'heading_variance'] = trip_data['heading'].std()
            
            if 'route_deviation_score' in trip_data.columns:
                trip_features.loc[idx, 'max_route_deviation'] = trip_data['route_deviation_score'].max()
                trip_features.loc[idx, 'avg_route_deviation'] = trip_data['route_deviation_score'].mean()
            
            if 'lane_deviation' in trip_data.columns:
                trip_features.loc[idx, 'lane_deviation_events'] = (trip_data['lane_deviation'] > 0.5).sum()
            
            # === DISTANCE & DURATION ===
            if 'trip_distance' in trip_data.columns:
                trip_features.loc[idx, 'total_distance'] = trip_data['trip_distance'].iloc[0]
            
            if 'trip_duration' in trip_data.columns:
                trip_features.loc[idx, 'duration_minutes'] = trip_data['trip_duration'].iloc[0]
            elif 'timestamp' in trip_data.columns:
                duration_sec = (trip_data['timestamp'].max() - trip_data['timestamp'].min()).total_seconds()
                trip_features.loc[idx, 'duration_minutes'] = duration_sec / 60
            
            # === TEMPORAL FEATURES ===
            if 'timestamp' in trip_data.columns:
                trip_features.loc[idx, 'hour_of_day'] = trip_data['timestamp'].dt.hour.mode()[0]
                trip_features.loc[idx, 'is_night'] = int(
                    trip_data['timestamp'].dt.hour.mode()[0] in range(20, 6)
                )
            
            # === BEHAVIORAL FEATURES ===
            if 'behavioral_consistency_index' in trip_data.columns:
                trip_features.loc[idx, 'behavior_consistency'] = trip_data['behavioral_consistency_index'].mean()
            
            if 'brake_usage' in trip_data.columns:
                trip_features.loc[idx, 'avg_brake_usage'] = trip_data['brake_usage'].mean()
            
            # === ENVIRONMENTAL CONTEXT ===
            if 'weather_conditions' in trip_data.columns:
                trip_features.loc[idx, 'weather_mode'] = trip_data['weather_conditions'].mode()[0]
            
            if 'traffic_condition' in trip_data.columns:
                trip_features.loc[idx, 'traffic_mode'] = trip_data['traffic_condition'].mode()[0]
            
            # === SAFETY INDICATORS ===
            if 'geofencing_violation' in trip_data.columns:
                trip_features.loc[idx, 'geofence_violations'] = trip_data['geofencing_violation'].sum()
            
            # === ENGINEERED RISK METRICS ===
            # Danger index: High speed + High deviation = High risk
            if 'route_deviation_score' in trip_data.columns:
                trip_features.loc[idx, 'danger_index'] = (
                    trip_data['route_deviation_score'] * trip_data['speed']
                ).mean()
            
            # Speed-distance efficiency
            if 'trip_distance' in trip_data.columns and trip_features.loc[idx, 'duration_minutes'] > 0:
                trip_features.loc[idx, 'speed_distance_ratio'] = (
                    trip_features.loc[idx, 'total_distance'] / trip_features.loc[idx, 'duration_minutes']
                )
        
        print(f"✓ Generated {len(trip_features.columns)} trip-level features for {len(trip_features)} trips")
        return trip_features
    
    def prepare_features(self, df, is_trip_aggregated=False):
        """
        Unified feature preparation supporting both raw and aggregated data
        
        Args:
            df: Input dataframe
            is_trip_aggregated: If True, df is already trip-level features
        """
        if not is_trip_aggregated and 'trip_id' in df.columns:
            # If raw GPS data, aggregate to trip level first
            df = self.engineer_trip_features(df)
        
        df = df.copy()
        
        # Define feature categories
        numerical_features = [
            'avg_speed', 'max_speed', 'min_speed', 'speed_variability', 'speed_changes',
            'stop_count', 'stop_ratio', 'lat_std', 'lon_std', 'bearing_variance',
            'heading_variance', 'total_distance', 'duration_minutes',
            'avg_acceleration', 'harsh_braking_count', 'harsh_accel_count',
            'prolonged_stops', 'max_route_deviation', 'avg_route_deviation',
            'lane_deviation_events', 'hour_of_day', 'is_night',
            'behavior_consistency', 'avg_brake_usage', 'geofence_violations',
            'danger_index', 'speed_distance_ratio'
        ]
        
        categorical_features = ['weather_mode', 'traffic_mode', 'weather_conditions', 
                               'road_type', 'traffic_condition']
        
        # Encode categoricals
        for cat_col in categorical_features:
            if cat_col in df.columns:
                if cat_col not in self.le_dict:
                    self.le_dict[cat_col] = LabelEncoder()
                    df[cat_col] = self.le_dict[cat_col].fit_transform(df[cat_col].astype(str))
                else:
                    df[cat_col] = self.le_dict[cat_col].transform(df[cat_col].astype(str))
        
        # Select available features
        all_features = [c for c in numerical_features + categorical_features if c in df.columns]
        
        # Handle missing values with median (robust to outliers)
        for col in all_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        
        self.feature_names = all_features
        
        print(f"✓ Prepared {len(all_features)} features: {all_features[:5]}...")
        return df[all_features]
    
    def train(self, df, is_trip_aggregated=False):
        """
        Train the complete anomaly detection system
        
        Returns:
            dict: Training results with predictions and metrics
        """
        print("\n" + "="*60)
        print("🚀 TRAINING ENHANCED AIOT ANOMALY DETECTION SYSTEM")
        print("="*60)
        
        # Feature preparation
        X = self.prepare_features(df, is_trip_aggregated)
        X_scaled = self.scaler.fit_transform(X)
        
        # === 1. ISOLATION FOREST (Unsupervised Anomaly Detection) ===
        print("\n📊 Training Isolation Forest...")
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,  # Increased for stability
            max_samples=0.8,
            n_jobs=-1
        )
        self.iso_forest.fit(X_scaled)
        
        iso_predictions = self.iso_forest.predict(X_scaled)  # -1 = anomaly, 1 = normal
        iso_scores = self.iso_forest.decision_function(X_scaled)
        
        print(f"✓ Isolation Forest detected {(iso_predictions == -1).sum()} anomalies")
        
        # === 2. DBSCAN CLUSTERING (Behavioral Pattern Detection) ===
        print("\n🔍 Running DBSCAN clustering...")
        self.dbscan = DBSCAN(eps=0.5, min_samples=3)
        clusters = self.dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = (clusters == -1).sum()
        
        print(f"✓ Found {n_clusters} behavioral clusters")
        print(f"✓ {n_noise} noise points (potential anomalies)")
        
        # Build cluster profiles
        for cluster_id in set(clusters):
            if cluster_id == -1:
                continue
            cluster_mask = clusters == cluster_id
            self.cluster_profiles[cluster_id] = {
                'size': cluster_mask.sum(),
                'centroid': X_scaled[cluster_mask].mean(axis=0)
            }
        
        # === 3. COMPILE RESULTS ===
        results_df = X.copy()
        results_df['iso_prediction'] = iso_predictions
        results_df['iso_score'] = iso_scores
        results_df['is_anomaly_IF'] = iso_predictions == -1
        results_df['cluster'] = clusters
        results_df['is_cluster_noise'] = clusters == -1
        
        # === 4. RULE-BASED DETECTION ===
        print("\n⚖️ Applying rule-based detection...")
        rule_alerts = self._apply_rules(results_df)
        print(f"✓ Generated {len(rule_alerts)} rule-based alerts")
        
        # === 5. HYBRID ANOMALY SCORING ===
        print("\n🎯 Computing hybrid anomaly scores...")
        final_results = self._compute_hybrid_scores(results_df, rule_alerts)
        
        print("\n" + "="*60)
        print("📈 TRAINING COMPLETE - SUMMARY")
        print("="*60)
        print(f"Total samples: {len(final_results)}")
        print(f"\nRisk Distribution:")
        print(final_results['risk_level'].value_counts())
        print(f"\nTop 3 Most Anomalous:")
        print(final_results.nlargest(3, 'final_anomaly_score')[
            ['avg_speed', 'stop_ratio', 'final_anomaly_score', 'risk_level']
        ])
        
        return {
            'results': final_results,
            'rule_alerts': rule_alerts,
            'n_anomalies': (final_results['risk_level'] == 'HIGH').sum(),
            'feature_names': self.feature_names
        }
    
    def _apply_rules(self, features_df):
        """Enhanced rule-based detection with Ghana-specific thresholds"""
        alerts = []
        
        for idx, trip in features_df.iterrows():
            # Rule 1: Excessive speed (Ghana speed limits: 50 urban, 100 highway)
            if 'max_speed' in features_df.columns and trip['max_speed'] > 120:
                alerts.append({
                    'trip_id': idx,
                    'rule': 'EXCESSIVE_SPEED',
                    'severity': 'HIGH',
                    'details': f"Max speed: {trip['max_speed']:.1f} km/h"
                })
            
            # Rule 2: Unusual stop pattern (forced stop indicator)
            if 'stop_ratio' in features_df.columns and 'stop_count' in features_df.columns:
                if trip['stop_ratio'] > 0.3 and trip['stop_count'] > 5:
                    alerts.append({
                        'trip_id': idx,
                        'rule': 'UNUSUAL_STOPS',
                        'severity': 'MEDIUM',
                        'details': f"Stop ratio: {trip['stop_ratio']:.2%}, Count: {trip['stop_count']}"
                    })
            
            # Rule 3: Erratic driving (speed variability)
            if 'speed_variability' in features_df.columns and 'avg_speed' in features_df.columns:
                if trip['speed_variability'] > trip['avg_speed'] * 0.5:
                    alerts.append({
                        'trip_id': idx,
                        'rule': 'ERRATIC_DRIVING',
                        'severity': 'MEDIUM',
                        'details': f"Speed std: {trip['speed_variability']:.1f}"
                    })
            
            # Rule 4: Route deviation
            variance_col = 'bearing_variance' if 'bearing_variance' in features_df.columns else 'heading_variance'
            if variance_col in features_df.columns:
                if trip[variance_col] > 90:
                    alerts.append({
                        'trip_id': idx,
                        'rule': 'ROUTE_DEVIATION',
                        'severity': 'HIGH',
                        'details': f"{variance_col}: {trip[variance_col]:.1f}°"
                    })
            
            # Rule 5: Night time risk (22:00 - 05:00 in Ghana)
            if 'hour_of_day' in features_df.columns:
                if trip['hour_of_day'] >= 22 or trip['hour_of_day'] <= 5:
                    if 'stop_ratio' in features_df.columns and trip['stop_ratio'] > 0.2:
                        alerts.append({
                            'trip_id': idx,
                            'rule': 'NIGHT_RISK',
                            'severity': 'MEDIUM',
                            'details': f"Late night trip with stops at {trip['hour_of_day']}:00"
                        })
            
            # Rule 6: Geofence violation
            if 'geofence_violations' in features_df.columns:
                if trip['geofence_violations'] > 0:
                    alerts.append({
                        'trip_id': idx,
                        'rule': 'GEOFENCE_VIOLATION',
                        'severity': 'HIGH',
                        'details': f"Entered unsafe zone {trip['geofence_violations']} times"
                    })
            
            # Rule 7: Dangerous behavior combo
            if 'danger_index' in features_df.columns:
                if trip['danger_index'] > 100:  # High speed + High deviation
                    alerts.append({
                        'trip_id': idx,
                        'rule': 'DANGER_COMBO',
                        'severity': 'HIGH',
                        'details': f"Danger index: {trip['danger_index']:.1f}"
                    })
        
        return pd.DataFrame(alerts) if alerts else pd.DataFrame()
    
    def _compute_hybrid_scores(self, features_df, rule_alerts, weights=None):
        """Combine ML, rules, and clustering for final risk assessment"""
        
        if weights is None:
            weights = {'ml': 0.6, 'rules': 0.3, 'cluster': 0.1}
        
        final_scores = []
        
        for idx, trip in features_df.iterrows():
            score = 0
            
            # Component 1: ML Score (Isolation Forest)
            ml_score = 1 / (1 + np.exp(trip['iso_score']))  # Sigmoid normalization
            score += weights['ml'] * ml_score
            
            # Component 2: Rule-based Score
            if len(rule_alerts) > 0:
                trip_alerts = rule_alerts[rule_alerts['trip_id'] == idx]
                if len(trip_alerts) > 0:
                    severity_map = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3}
                    rule_score = min(1.0, sum(severity_map.get(s, 0) for s in trip_alerts['severity']) / 2)
                    score += weights['rules'] * rule_score
            
            # Component 3: Cluster Distance Score
            if trip['is_cluster_noise']:
                score += weights['cluster'] * 1.0
            
            final_scores.append(score)
        
        features_df['final_anomaly_score'] = final_scores
        features_df['risk_level'] = pd.cut(
            final_scores,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        return features_df
    
    def predict(self, df, is_trip_aggregated=False):
        """Real-time prediction for new trips"""
        X = self.prepare_features(df, is_trip_aggregated)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all components
        iso_predictions = self.iso_forest.predict(X_scaled)
        iso_scores = self.iso_forest.decision_function(X_scaled)
        clusters = self.dbscan.fit_predict(X_scaled)
        
        results_df = X.copy()
        results_df['iso_prediction'] = iso_predictions
        results_df['iso_score'] = iso_scores
        results_df['cluster'] = clusters
        results_df['is_cluster_noise'] = clusters == -1
        
        rule_alerts = self._apply_rules(results_df)
        final_results = self._compute_hybrid_scores(results_df, rule_alerts)
        
        return final_results, rule_alerts
    
    def save_model(self, filepath='models/aiot_detector.pkl'):
        """Save trained model and preprocessors"""
        model_package = {
            'iso_forest': self.iso_forest,
            'dbscan': self.dbscan,
            'scaler': self.scaler,
            'le_dict': self.le_dict,
            'feature_names': self.feature_names,
            'cluster_profiles': self.cluster_profiles,
            'contamination': self.contamination
        }
        joblib.dump(model_package, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/aiot_detector.pkl'):
        """Load pre-trained model"""
        model_package = joblib.load(filepath)
        self.iso_forest = model_package['iso_forest']
        self.dbscan = model_package['dbscan']
        self.scaler = model_package['scaler']
        self.le_dict = model_package['le_dict']
        self.feature_names = model_package['feature_names']
        self.cluster_profiles = model_package['cluster_profiles']
        self.contamination = model_package['contamination']
        print(f"✓ Model loaded from {filepath}")


# =============================================================================
# EXAMPLE USAGE & PERFORMANCE EVALUATION
# =============================================================================

def evaluate_model_performance(model, X_test_scaled):
    """Evaluate inference speed and resource usage"""
    import time
    
    print("\n" + "="*60)
    print("⚡ PERFORMANCE EVALUATION")
    print("="*60)
    
    # Latency test
    start_time = time.time()
    for _ in range(100):
        sample = X_test_scaled[0].reshape(1, -1)
        model.iso_forest.predict(sample)
    inference_time = (time.time() - start_time) / 100
    
    print(f"Average inference time: {inference_time*1000:.2f} ms")
    print(f"Meets <5s NFR1 requirement: {'✅ YES' if inference_time < 5 else '❌ NO'}")
    print(f"Real-time capable (< 100ms): {'✅ YES' if inference_time < 0.1 else '⚠️  ACCEPTABLE'}")
    
    return inference_time


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AIoT PASSENGER SAFETY ANOMALY DETECTION SYSTEM")
    print("Ghana Ride-Hailing Safety Project")
    print("="*60)
    
    # Load your dataset
    df = pd.read_csv('your_trip_data.csv')
    
    print(f"\n📂 Loaded {len(df)} records from {df['trip_id'].nunique()} trips")
    
    # Initialize detector
    detector = EnhancedAIoTAnomalyDetector(
        contamination=0.05,  # Expect 5% anomalies
        random_state=42
    )
    
    # Train the system
    training_results = detector.train(df, is_trip_aggregated=False)
    
    # Evaluate performance
    X_test = detector.prepare_features(df)
    X_test_scaled = detector.scaler.transform(X_test)
    evaluate_model_performance(detector, X_test_scaled)
    
    # Save model for deployment
    detector.save_model('models/aiot_detector_v1.pkl')
    
    print("\n✅ Training complete! Model ready for API deployment.")
    print("📊 Check training_results['results'] for detailed analysis")