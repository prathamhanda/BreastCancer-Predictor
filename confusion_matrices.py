import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import os
import joblib

# Create output directory if it doesn't exist
output_dir = 'confusion_matrices'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and prepare data
print("Loading data...")
data = pd.read_csv("data/data.csv")
data = data.drop(['Unnamed: 32', 'id'], axis=1)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Train Random Forest
print("Training Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_scaled, y_train)
rf_predictions = rf.predict(X_test_scaled)

# Train SVM
print("Training SVM model...")
svm = SVC(
    C=1.0,
    kernel='rbf',
    probability=True,
    random_state=42,
    cache_size=2000
)
svm.fit(X_train_scaled, y_train)
svm_predictions = svm.predict(X_test_scaled)

# Train Logistic Regression
print("Training Logistic Regression model...")
lr = LogisticRegression(
    C=1.0,
    max_iter=500,
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train_scaled, y_train)
lr_predictions = lr.predict(X_test_scaled)

# Create confusion matrices
def generate_and_save_cm(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name}')
    
    labels = ['Benign (0)', 'Malignant (1)']
    plt.xticks([0.5, 1.5], labels)
    plt.yticks([0.5, 1.5], labels)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Print classification report
    report = classification_report(y_true, y_pred)
    with open(f'{output_dir}/{model_name}_classification_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n{model_name} Classification Report:")
    print(report)

# Generate and save confusion matrices
print("\nGenerating confusion matrices...")
generate_and_save_cm(y_test, rf_predictions, "Random_Forest")
generate_and_save_cm(y_test, svm_predictions, "SVM")
generate_and_save_cm(y_test, lr_predictions, "Logistic_Regression")

# Create a comparison table
accuracy_data = {
    'Model': ['Random Forest', 'SVM', 'Logistic Regression'],
    'Accuracy': [
        rf.score(X_test_scaled, y_test),
        svm.score(X_test_scaled, y_test),
        lr.score(X_test_scaled, y_test)
    ]
}

# Create and save comparison bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=pd.DataFrame(accuracy_data))
plt.title('Model Accuracy Comparison')
plt.ylim(0.8, 1.0)  # Adjust as needed
plt.tight_layout()
plt.savefig(f'{output_dir}/model_accuracy_comparison.png', dpi=300)
plt.close()

print(f"\nAll confusion matrices and reports saved to '{output_dir}' folder.")
print(f"Files generated:")
for file in os.listdir(output_dir):
    print(f"- {file}") 