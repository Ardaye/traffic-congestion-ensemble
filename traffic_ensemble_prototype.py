import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import warnings
warnings.filterwarnings('ignore')

# --- Core ML Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score

# --- Ensemble Stacking Imports ---
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

# --- The 3 Base Models ---
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# --- Geospatial & Routing ---
from geopy.distance import geodesic
import networkx as nx

# --- Visualization (Optional) ---
try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Folium not installed. Map visualization skipped.")


# ==============================================
# 1. DATA GENERATION (Mocks your dataset schema)
# ==============================================
def generate_mock_data(n=1000):
    np.random.seed(42)
    random.seed(42)
    
    event_types = ['Political Rally', 'Festival', 'Sports Event', 'Accident', 'Construction', 'Sudden Gathering']
    zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D']
    corridors = ['NH-1', 'MG Road', 'Ring Road', 'Main Bypass']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    statuses = ['Active', 'Resolved']
    vehicle_types = ['Car', 'Truck', 'Bus', 'Bike', 'None']
    
    data = []
    start_base = datetime(2025, 1, 1, 0, 0)
    
    for i in range(n):
        lat = 28.61 + np.random.uniform(-0.05, 0.05)
        lon = 77.23 + np.random.uniform(-0.05, 0.05)
        end_lat = lat + np.random.uniform(-0.01, 0.01)
        end_lon = lon + np.random.uniform(-0.01, 0.01)
        
        hours_offset = np.random.randint(0, 30*24)
        start_dt = start_base + timedelta(hours=hours_offset)
        duration_mins = np.random.randint(30, 360) 
        end_dt = start_dt + timedelta(minutes=duration_mins)
        
        if random.random() > 0.6:
            lead_time = np.random.randint(60, 720)  # Planned
        else:
            lead_time = np.random.randint(5, 45)    # Unplanned
        created_dt = start_dt - timedelta(minutes=lead_time)
        
        police_count = np.random.randint(1, 15)
        closure = random.choice([True, False])
        if closure:
            police_count += np.random.randint(2, 8)
            
        row = {
            'id': f'E{i:04d}',
            'event_type': random.choice(event_types),
            'latitude': lat,
            'longitude': lon,
            'endlatitude': end_lat,
            'endlongitude': end_lon,
            'address': f'{i} Main St, City',
            'end_address': f'{i} End St, City',
            'event_cause': random.choice(['Traffic', 'Crowd', 'VIP Movement', 'Infrastructure']),
            'requires_road_closure': closure,
            'start_datetime': start_dt,
            'end_datetime': end_dt,
            'status': random.choice(statuses),
            'authenticated': random.choice([True, False]),
            'modified_datetime': start_dt + timedelta(minutes=10),
            'map_file': f'map_{i}.pdf',
            'direction': random.choice(['Both', 'North', 'South', 'East', 'West']),
            'description': f"Mock event {i}",
            'veh_type': random.choice(vehicle_types),
            'veh_no': f'DL-{i}',
            'corridor': random.choice(corridors),
            'priority': random.choice(priorities),
            'cargo_material': random.choice(['None', 'Fuel', 'Perishable', 'Electronics']),
            'reason_breakdown': random.choice(['None', 'Engine', 'Tire', 'Accident']),
            'age_of_truck': np.random.randint(0, 15),
            'created_date': created_dt,
            'route_path': f"POLYLINE({lat} {lon}, {end_lat} {end_lon})",
            'client_id': f'C{np.random.randint(1,100)}',
            'created_by_id': f'U{np.random.randint(1,50)}',
            'last_modified_by_id': f'U{np.random.randint(1,50)}',
            'assigned_to_police_id': [f'P{np.random.randint(100,999)}' for _ in range(police_count)],
            'citizen_accident_id': f'CA{np.random.randint(1,100)}' if random.random()>0.8 else None,
            'comment': random.choice(['Heavy rain', 'VIP delayed', 'Crowd dispersed quickly', 'N/A']),
            'police_station': f'PS_{random.choice(["North", "South", "East", "West"])}',
            'meta_data': json.dumps({"weather": random.choice(["sunny", "rainy", "foggy"])}),
            'kgid': f'KG{i}',
            'resolved_at_address': f'Resolved {i}',
            'resolved_at_latitude': end_lat + 0.001,
            'resolved_at_longitude': end_lon + 0.001,
            'closed_by_id': f'C{i}' if random.random()>0.3 else None,
            'closed_datetime': end_dt + timedelta(minutes=5),
            'resolved_by_id': f'R{i}',
            'resolved_datetime': end_dt + timedelta(minutes=10),
            'gba_identifier': f'GBA{i}',
            'zone': random.choice(zones),
            'junction': random.choice(['J1', 'J2', 'J3', 'J4', 'J5'])
        }
        data.append(row)
        
    return pd.DataFrame(data)

# ==============================================
# 2. FEATURE ENGINEERING
# ==============================================
def engineer_features(df):
    """Extract derived features from raw dataset."""
    df = df.copy()
    
    # 1. Affected Road Length (km)
    df['affected_road_km'] = df.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), 
                             (row['endlatitude'], row['endlongitude'])).km, axis=1)
    df['affected_road_km'] = df['affected_road_km'].clip(lower=0.1)
    
    # 2. Temporal features
    df['start_hour'] = df['start_datetime'].dt.hour
    df['start_day_of_week'] = df['start_datetime'].dt.dayofweek
    df['is_weekend'] = df['start_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_peak_hour'] = df['start_hour'].apply(lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 20) else 0)
    
    # 3. Lead Time (hours)
    df['lead_time_hours'] = (df['start_datetime'] - df['created_date']).dt.total_seconds() / 3600.0
    df['is_planned'] = df['lead_time_hours'].apply(lambda x: 1 if x > 1 else 0)
    
    # 4. Target Variable: Duration (minutes) - Proxy for Impact
    df['duration_minutes'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
    df['duration_minutes'] = df['duration_minutes'].clip(upper=600)
    
    # 5. Manpower Count (Number of unique officers assigned)
    df['manpower_count'] = df['assigned_to_police_id'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    return df

# ==============================================
# 3. TRAINING THE ENSEMBLE (XGB + LGB + CatBoost)
# ==============================================
def train_models(df_processed):
    """Train a Stacking Ensemble of XGBoost, LightGBM, and CatBoost."""
    
    # Encode categorical variables
    le_event = LabelEncoder()
    le_zone = LabelEncoder()
    le_corridor = LabelEncoder()
    le_priority = LabelEncoder()
    
    df_processed['event_type_enc'] = le_event.fit_transform(df_processed['event_type'])
    df_processed['zone_enc'] = le_zone.fit_transform(df_processed['zone'])
    df_processed['corridor_enc'] = le_corridor.fit_transform(df_processed['corridor'])
    df_processed['priority_enc'] = le_priority.fit_transform(df_processed['priority'])
    
    # Define Features
    numerical_cols = ['affected_road_km', 'start_hour', 'is_weekend', 'is_peak_hour', 
                      'lead_time_hours', 'is_planned', 'age_of_truck']
    feature_cols = numerical_cols + ['event_type_enc', 'zone_enc', 'corridor_enc', 'priority_enc']
    
    X = df_processed[feature_cols]
    y_reg = df_processed['duration_minutes']      # Target 1: Duration
    y_cls = df_processed['requires_road_closure'] # Target 2: Closure (Binary)
    
    # Train-Test Split
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )
    
    # ==============================================
    # A. STACKING REGRESSOR (Duration Prediction)
    # ==============================================
    print("🔥 Building Ensemble Regressor (XGB + LGB + CatBoost)...")
    base_regressors = [
        ('lightgbm', LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, verbose=-1, random_state=42)),
        ('xgboost', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, verbosity=0, random_state=42)),
        ('catboost', CatBoostRegressor(n_estimators=100, learning_rate=0.1, depth=5, verbose=0, random_state=42))
    ]
    # Linear Regression as the meta-model for regression
    meta_regressor = LinearRegression()
    
    ensemble_reg = StackingRegressor(
        estimators=base_regressors,
        final_estimator=meta_regressor,
        cv=5  # 5-fold cross-validation to generate training predictions for the meta-model
    )
    ensemble_reg.fit(X_train, y_reg_train)
    reg_pred = ensemble_reg.predict(X_test)
    reg_mae = mean_absolute_error(y_reg_test, reg_pred)
    
    # ==============================================
    # B. STACKING CLASSIFIER (Closure Prediction)
    # ==============================================
    print("🔥 Building Ensemble Classifier (XGB + LGB + CatBoost)...")
    base_classifiers = [
        ('lightgbm', LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, verbose=-1, random_state=42)),
        ('xgboost', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, verbosity=0, random_state=42)),
        ('catboost', CatBoostClassifier(n_estimators=100, learning_rate=0.1, depth=5, verbose=0, random_state=42))
    ]
    # Logistic Regression as the meta-model for classification
    meta_classifier = LogisticRegression()
    
    ensemble_cls = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=meta_classifier,
        cv=5
    )
    ensemble_cls.fit(X_train, y_cls_train)
    cls_pred = ensemble_cls.predict(X_test)
    cls_acc = accuracy_score(y_cls_test, cls_pred)
    
    print("="*50)
    print("✅ ENSEMBLE MODEL PERFORMANCE (Stacking)")
    print(f"   Duration Prediction MAE: {reg_mae:.2f} minutes")
    print(f"   Road Closure Prediction Accuracy: {cls_acc:.2f}")
    print("="*50)
    
    # Return models + encoders (same interface for the rest of the pipeline)
    return ensemble_reg, ensemble_cls, le_event, le_zone, le_corridor, le_priority, feature_cols

# ==============================================
# 4. RECOMMENDATION ENGINE (Manpower, Barricades, Diversion)
# ==============================================
def make_recommendation(new_event, df_historical, regressor, classifier, 
                        le_event, le_zone, le_corridor, le_priority, feature_cols):
    """Predict impact and recommend resources/diversions for a new event."""
    
    # Feature Engineering for the new event
    new_df = pd.DataFrame([new_event])
    new_df = engineer_features(new_df)
    
    # Encode categoricals using fitted encoders
    try:
        new_df['event_type_enc'] = le_event.transform(new_df['event_type'])
        new_df['zone_enc'] = le_zone.transform(new_df['zone'])
        new_df['corridor_enc'] = le_corridor.transform(new_df['corridor'])
        new_df['priority_enc'] = le_priority.transform(new_df['priority'])
    except ValueError:
        print("Warning: Unknown category in new event. Defaulting to 0.")
        new_df['event_type_enc'] = 0
        new_df['zone_enc'] = 0
        new_df['corridor_enc'] = 0
        new_df['priority_enc'] = 0
    
    # Predict using the Ensemble
    X_new = new_df[feature_cols]
    pred_duration = regressor.predict(X_new)[0]
    prob_closure = classifier.predict_proba(X_new)[0][1]
    
    # --- Manpower Recommendation ---
    avg_manpower = df_historical[
        (df_historical['event_type'] == new_event['event_type']) & 
        (df_historical['zone'] == new_event['zone'])
    ]['manpower_count'].mean()
    if pd.isna(avg_manpower):
        avg_manpower = df_historical['manpower_count'].mean()
    
    adjustment = 1 + (pred_duration / 120) * 0.2
    recommended_manpower = int(avg_manpower * adjustment * (1 + prob_closure * 0.5))
    recommended_manpower = max(2, min(25, recommended_manpower))
    
    # --- Barricading Recommendation ---
    road_km = new_df['affected_road_km'].values[0]
    base_barricade_m = road_km * 1000
    if prob_closure > 0.7:
        barricade_meters = base_barricade_m * 1.5
    else:
        barricade_meters = base_barricade_m * 0.8
    barricade_meters = max(50, int(barricade_meters))
    
    # --- Diversion Planning (Grid Simulation) ---
    G = nx.grid_2d_graph(10, 10)
    lat_offset, lon_offset = 28.61, 77.23
    scale = 100
    try:
        x = int((new_event['longitude'] - lon_offset) * scale) % 10
        y = int((new_event['latitude'] - lat_offset) * scale) % 10
    except:
        x, y = 5, 5
    blocked_node = (x, y)
    
    G_copy = G.copy()
    if blocked_node in G_copy.nodes:
        G_copy.remove_node(blocked_node)
    
    diversion_path = None
    try:
        diversion_path = nx.shortest_path(G_copy, source=(0,0), target=(9,9))
    except nx.NetworkXNoPath:
        diversion_path = ["No alternate path found."]
    
    return {
        "event_id": new_event['id'],
        "predicted_duration_minutes": round(pred_duration, 1),
        "road_closure_probability": round(prob_closure * 100, 1),
        "recommended_manpower": recommended_manpower,
        "barricade_meters": barricade_meters,
        "blocked_location": blocked_node,
        "diversion_route": diversion_path[:5] if diversion_path else []
    }

# ==============================================
# 5. POST-EVENT LEARNING SYSTEM
# ==============================================
class LearningSystem:
    def __init__(self):
        self.errors = []
        
    def log_prediction(self, event_id, predicted_duration, actual_duration, 
                       predicted_manpower, actual_manpower):
        self.errors.append({
            'event_id': event_id,
            'dur_error': actual_duration - predicted_duration,
            'man_error': actual_manpower - predicted_manpower,
            'timestamp': datetime.now()
        })
        print(f"\n[LEARNING] Logged error for {event_id}. Duration off by {actual_duration - predicted_duration:.1f} mins.")
        
    def get_feedback(self):
        if not self.errors:
            return "No errors logged yet."
        df_err = pd.DataFrame(self.errors)
        return {
            "avg_duration_bias": round(df_err['dur_error'].mean(), 2),
            "avg_manpower_bias": round(df_err['man_error'].mean(), 2),
            "total_events": len(self.errors)
        }

# ==============================================
# 6. MAIN EXECUTION
# ==============================================
if __name__ == "__main__":
    print("🚦 INITIALIZING TRAFFIC CONGESTION PROTOTYPE (ENSEMBLE EDITION)...")
    
    # A. Generate & Process Data
    raw_df = generate_mock_data(1000)
    processed_df = engineer_features(raw_df)
    
    # B. Train the Stacking Ensemble
    (regressor, classifier, le_event, le_zone, le_corridor, 
     le_priority, feature_cols) = train_models(processed_df)
    
    # C. Simulate a New Event
    new_event = {
        'id': 'E_NEW_001',
        'event_type': 'Political Rally',
        'zone': 'Zone_A',
        'corridor': 'NH-1',
        'latitude': 28.62,
        'longitude': 77.24,
        'endlatitude': 28.625,
        'endlongitude': 77.245,
        'priority': 'High',
        'start_datetime': datetime(2025, 6, 18, 18, 0),  # Peak hour
        'end_datetime': datetime(2025, 6, 18, 20, 0),
        'created_date': datetime(2025, 6, 18, 10, 0),    # Planned
        'age_of_truck': 0,
        'requires_road_closure': False,
        'assigned_to_police_id': [],
        'status': 'Active'
    }
    
    print("\n📡 RECEIVED NEW EVENT:")
    print(f"   {new_event['event_type']} in {new_event['zone']} at {new_event['start_datetime']}")
    
    # D. Get Recommendations from the Ensemble
    recommendation = make_recommendation(
        new_event, processed_df, regressor, classifier, 
        le_event, le_zone, le_corridor, le_priority, feature_cols
    )
    
    print("\n🛠️ RECOMMENDATION OUTPUT:")
    print(f"   ➤ Predicted Impact Duration: {recommendation['predicted_duration_minutes']} mins")
    print(f"   ➤ Road Closure Probability: {recommendation['road_closure_probability']}%")
    print(f"   ➤ Deploy Officers: {recommendation['recommended_manpower']} personnel")
    print(f"   ➤ Barricading Required: {recommendation['barricade_meters']} meters")
    print(f"   ➤ Alternate Diversion Path (first 5 nodes): {recommendation['diversion_route']}")
    
    # E. Simulate Post-Event (Actual results)
    actual_duration = 95.0   # Let's say it took 95 mins
    actual_manpower = 14     # They deployed 14 officers
    
    # F. Learning System logs the discrepancy
    learning_sys = LearningSystem()
    learning_sys.log_prediction(
        event_id=new_event['id'],
        predicted_duration=recommendation['predicted_duration_minutes'],
        actual_duration=actual_duration,
        predicted_manpower=recommendation['recommended_manpower'],
        actual_manpower=actual_manpower
    )
    
    # G. Show Learning Feedback
    print("\n📊 POST-EVENT LEARNING INSIGHT:")
    feedback = learning_sys.get_feedback()
    print(f"   ➤ Model Bias (Under/Over estimation): {feedback['avg_duration_bias']} minutes")
    print(f"   ➤ Manpower Allocation Bias: {feedback['avg_manpower_bias']} officers")
    print("   ➤ (If bias > threshold, auto-retraining would trigger here in production).")
    
    # H. Optional Map
    if FOLIUM_AVAILABLE:
        print("\n🗺️ Generating Diversion Map... (Check 'diversion_map.html')")
        m = folium.Map(location=[28.61, 77.23], zoom_start=13)
        folium.Marker(
            location=[new_event['latitude'], new_event['longitude']],
            popup='🚧 Blocked (Event)',
            icon=folium.Icon(color='red')
        ).add_to(m)
        path_points = [(28.61, 77.23), (28.615, 77.25), (28.62, 77.27)]
        folium.PolyLine(path_points, color="green", weight=5, popup="Diversion Route").add_to(m)
        m.save("diversion_map.html")
        print("   Map saved! Open 'diversion_map.html' in your browser.")
    
    print("\n✅ ENSEMBLE PROTOTYPE EXECUTION COMPLETE.")