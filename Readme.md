🚚 Flippin' Traffic: Predictive Congestion Management for Last-Mile Logistics
Our Submission for the Flipkart Gridlock Hackathon
📌 The Problem We're Solving
Flipkart's delivery ecosystem relies heavily on predictable movement. But cities are chaotic.

One political rally or an unexpected road closure can cascade into a nightmare for delivery executives—delayed packages, wasted fuel, missed SLAs, and frustrated customers.

Currently, there is no system that quantifies the actual impact of these events in advance. Resource deployment is pure guesswork, and there's zero learning from past incidents.

Our goal: Turn this unpredictability into a data-driven, proactive strategy.

💡 Our Solution
We built an intelligent system that ingests real-world event data and outputs real-time, actionable recommendations specifically designed for logistics optimization.

Here's exactly what we deliver:

Impact Forecast: Predicts exactly how many minutes traffic will be disrupted.

Closure Probability: Tells you the likelihood of a complete roadblock.

Dynamic Rerouting: Instantly suggests alternate paths to redirect delivery fleets.

Resource Planning: Recommends how many personnel/barricades are needed.

Explainability: Shows which factors (peak hour, road length, zone) drive each prediction.

Operational Metrics: SLA adherence and MAPE for real-world business tracking.

Most importantly, it learns. The system logs prediction errors and simulates retraining, providing the foundation for continuous model improvement.

🏗️ How It Works (Under the Hood)
1. Real Data Pipeline
We load real anonymized incident data from the provided CSV (8,173 rows), clean it, and handle missing values:

Parses coordinates and calculates affected road length using Haversine distance

Engineers temporal features (hour-of-day, peak hour flags, weekend indicators)

Calculates lead time (planned vs. unplanned events)

Extracts historical manpower counts from past deployments

2. Time-Series Validation
The model uses chronological train-test splitting (80% train, 20% test) to prevent future data leakage. Cross-validation is performed using TimeSeriesSplit to respect temporal order.

3. Stacking Ensemble
Instead of relying on one model, we combine three state-of-the-art algorithms:

LightGBM: Fast, leaf-wise growth for complex interactions

XGBoost: Level-wise growth with regularization for robust performance

CatBoost: Excellent handling of categorical variables

These are stacked with meta-models (Linear Regression for regression, Logistic Regression for classification) using 5-fold cross-validation.

4. Optimization-Based Resource Allocation
Resources are allocated using a cost-minimization heuristic:

Manpower: Minimum 2 officers, scaled with predicted duration

Barricades: Based on road length and closure probability

A buffer is added to account for prediction uncertainty

5. Diversion Planning
The system attempts to fetch real road networks from OpenStreetMap (via OSMnx) and finds the shortest alternate path around the blocked area. If OSMnx is unavailable, it gracefully falls back to a grid-based simulation.

6. Post-Event Learning Loop
Once an event ends, the system logs the discrepancy between predicted and actual duration/manpower. When 5 errors are accumulated, it simulates retraining—building the foundation for automatic model improvement.

📊 Model Evaluation Metrics
The system computes comprehensive metrics on unseen test data:

Regression	Classification
MAE	Accuracy
RMSE	Precision
R²	Recall
MAPE	F1-Score
SLA Adherence	ROC-AUC
🖥️ Sample Terminal Output
Here's the complete prototype in action:

text
ℹ️ OSMnx not installed. Using grid-based diversion fallback.
🚦 INITIALIZING TRAFFIC CONGESTION PROTOTYPE (HACKATHON EDITION)...
📂 Loading REAL dataset from: Astram event data_anonymized - Astram event data_anonymizedb40ac87 (1).csv
   Original rows: 8173
   Columns: ['id', 'event_type', 'latitude', 'longitude', 'endlatitude', 'endlongitude', 'address', 'end_address', 'event_cause', 'requires_road_closure', 'start_datetime', 'end_datetime', 'status', 'authenticated', 'modified_datetime', 'map_file', 'direction', 'description', 'veh_type', 'veh_no', 'corridor', 'priority', 'cargo_material', 'reason_breakdown', 'age_of_truck', 'created_date', 'route_path', 'client_id', 'created_by_id', 'last_modified_by_id', 'assigned_to_police_id', 'citizen_accident_id', 'comment', 'police_station', 'meta_data', 'kgid', 'resolved_at_address', 'resolved_at_latitude', 'resolved_at_longitude', 'closed_by_id', 'closed_datetime', 'resolved_by_id', 'resolved_datetime', 'gba_identifier', 'zone', 'junction']
   Dropped 169 rows with invalid coordinates
   Dropped 104 rows with missing start_datetime
   Dropped 7535 rows with missing end_datetime
   ✅ 365 valid rows loaded!
   ✅ Feature engineering done. 365 rows.
📊 TimeSeries CV MAE: 209.41 mins
📊 TimeSeries CV Accuracy: 0.98
🔥 Training Final Ensemble on Full Dataset...
==================================================
✅ ENSEMBLE TRAINED (Time-Series Aware)
   Train MAE: 107.16 mins
   Train Acc: 1.00
==================================================

📡 RECEIVED NEW EVENT:
   Political Rally in Zone_A at 2025-06-18 18:00:00
   ✅ Feature engineering done. 1 rows.

🗺️ Fetching diversion routes...
   Route found with 19 nodes.

🔍 EXPLAINABILITY: Feature contributions to prediction:
   - affected_road_km: +72.36 mins
   - start_hour: +3.10 mins
   - is_weekend: +1.94 mins
   - is_peak_hour: +36.61 mins
   - lead_time_hours: +25.57 mins
   - is_planned: +0.00 mins
   - age_of_truck: +0.00 mins
   - event_type_enc: -1.73 mins
   - zone_enc: +22.21 mins
   - corridor_enc: +44.64 mins
   - priority_enc: +1.51 mins

🛠️ OPTIMIZED RECOMMENDATION OUTPUT:
   ➤ Predicted Impact Duration: 341.7 mins
   ➤ Road Closure Probability: 94.8%
   ➤ [OPTIMIZED] Deploy Officers: 4 personnel
   ➤ [OPTIMIZED] Barricading Required: 1080 meters
   ➤ Diversion Route: Grid-based fallback

[LEARNING] Logged actual duration: 95.0 mins.

📊 OPERATIONAL METRICS:
   ➤ SLA Adherence (±30 mins): ❌ FAIL
   ➤ MAPE: 259.6%

🗺️ Diversion map saved as 'diversion_map.html'

✅ HACKATHON PROTOTYPE EXECUTION COMPLETE.
🗺️ Visual Output
The script generates an interactive HTML map (diversion_map.html):

Red Marker: The blocked location (event)

Green PolyLine: The recommended alternate route

This gives delivery command centers a quick, visual understanding of exactly where the disruption is and how to route their executives around it.

<img width="485" height="278" alt="image" src="https://github.com/user-attachments/assets/8bd0f7f9-e1ed-46f3-bbe1-19e20ce5d5b5" />
🧰 Tech Stack
Category	Libraries
Data Processing	Pandas, NumPy, Geopy
Machine Learning	Scikit-learn (Stacking), LightGBM, XGBoost, CatBoost
Routing	NetworkX, OSMnx (optional)
Explainability	SHAP
Visualization	Folium (Interactive Maps)
⚙️ How to Run
1. Clone the Repository
bash
git clone <repo-url>
cd <repo-folder>
2. Install Dependencies
bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost networkx geopy folium shap
Note: OSMnx is optional. If not installed, the system gracefully falls back to grid-based routing.

3. Place Your Dataset
Ensure your CSV file is named Astram event data_anonymized - Astram event data_anonymizedb40ac87 (1).csv (or update the path in the code).

4. Run the Prototype
bash
python traffic_ensemble_prototype.py
🚀 Future Scope
1.Live Traffic Integration: Connect to Google Maps/TomTom APIs for real-time speeds

2.Automatic Retraining: Implement actual .fit() on accumulated errors

3.Real Road Networks: Replace grid fallback with OSMnx for production

4.Web Dashboard: Build a React/Flask dashboard for command centers

5.Mobile Integration: Push alerts to delivery executives' phones

6.Historical Learning: Store errors in a database for continuous improvement

