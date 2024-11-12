# app.py

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your data
#data = pd.read_csv("ball_by_ball_ipl.csv")

# Define Flask app
app = Flask(__name__)

def get_top_11_players(venue, opposition_team):
    filtered_data = data[(data['Venue'] == venue) & (data['Bat Second'] == opposition_team)]
    
    if filtered_data.empty:
        return []

    # Encode categorical data
    label_encoders = {}
    categorical_features = ['Venue', 'Bat Second']
    for feature in categorical_features:
        le = LabelEncoder()
        filtered_data[feature] = le.fit_transform(filtered_data[feature].astype(str))
        label_encoders[feature] = le
    
    # Aggregate stats
    aggregated_data = filtered_data.groupby('Batter').agg({
        'Batter Runs': 'mean',          
        'Batter Balls Faced': 'mean',    
        'Bowler Runs Conceded': 'mean',   
        'Valid Ball': 'mean',             
        'Venue': 'mean'                   
    }).reset_index()
    
    # Check required features
    features = ['Batter Runs', 'Batter Balls Faced', 'Bowler Runs Conceded', 'Valid Ball', 'Venue']
    for feature in features:
        if feature not in aggregated_data.columns:
            raise KeyError(f"Feature '{feature}' is not present in the aggregated data.")

    X = aggregated_data[features]
    y = np.random.choice([0, 1], size=(len(aggregated_data),), p=[0.8, 0.2])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities and get top 11 players
    predicted_probabilities = model.predict_proba(X_scaled)[:, 1]
    aggregated_data['Predicted Probability'] = predicted_probabilities
    top_11_players = aggregated_data.sort_values(by='Predicted Probability', ascending=False).head(11)

    return top_11_players[['Batter', 'Predicted Probability']].to_dict(orient='records')

@app.route('/top_11_players', methods=['POST'])
def top_11_players_endpoint():
    data = request.json
    venue = data.get('venue')
    opposition_team = data.get('opposition_team')
    if not venue or not opposition_team:
        return jsonify({"error": "Both 'venue' and 'opposition_team' are required."}), 400
    
    top_11_players = get_top_11_players(venue, opposition_team)
    return jsonify(top_11_players)

if __name__ == '__main__':
    app.run(debug=True)
