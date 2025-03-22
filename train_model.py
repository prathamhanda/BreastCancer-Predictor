import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load and prepare data
data = pd.read_csv("data\\data.csv")
data = data.drop(['Unnamed: 32', 'id'], axis=1)
data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

# Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler using joblib
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Print accuracy
print(f"Model accuracy: {model.score(X_test_scaled, y_test):.2%}")
print("Model and scaler saved successfully!") 