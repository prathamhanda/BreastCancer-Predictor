import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import joblib
import os
from concurrent.futures import ThreadPoolExecutor

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load and prepare data
print("Loading data...")
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
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

def train_random_forest():
    rf = RandomForestClassifier(
        n_estimators=100,  # Reduced from 200
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    return rf, rf.score(X_test_scaled, y_test)

def train_svm():
    svm = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=42,
        cache_size=2000  # Increase cache size for faster computation
    )
    svm.fit(X_train_scaled, y_train)
    return svm, svm.score(X_test_scaled, y_test)

def train_logistic_regression():
    lr = LogisticRegression(
        C=1.0,
        max_iter=500,  # Reduced from 1000
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    lr.fit(X_train_scaled, y_train)
    return lr, lr.score(X_test_scaled, y_test)

# Train models in parallel
print("Training models...")
with ThreadPoolExecutor() as executor:
    rf_future = executor.submit(train_random_forest)
    svm_future = executor.submit(train_svm)
    lr_future = executor.submit(train_logistic_regression)
    
    # Get results
    rf_model, rf_accuracy = rf_future.result()
    svm_model, svm_accuracy = svm_future.result()
    lr_model, lr_accuracy = lr_future.result()

# Create and train voting classifier
print("Creating ensemble model...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('svm', svm_model),
        ('lr', lr_model)
    ],
    voting='soft',
    weights=[2, 2, 1]
)

voting_clf.fit(X_train_scaled, y_train)
voting_accuracy = voting_clf.score(X_test_scaled, y_test)

# Show accuracies
print("\nMODEL ACCURACIES")
print("-" * 30)
print(f"Random Forest: {rf_accuracy:.2%}")
print(f"SVM: {svm_accuracy:.2%}")
print(f"Logistic Regression: {lr_accuracy:.2%}")
print(f"Ensemble Model: {voting_accuracy:.2%}")
print("-" * 30)

# Save ensemble model and scaler
print("Saving models...")
joblib.dump(voting_clf, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("Done!")
