import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


def create_model(data): 
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create base models
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    
    svm = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=42,
        cache_size=2000
    )
    
    lr = LogisticRegression(
        C=1.0,
        max_iter=500,
        random_state=42,
        n_jobs=-1
    )
    
    # Create and train ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('svm', svm),
            ('lr', lr)
        ],
        voting='soft',
        weights=[2, 2, 1]
    )
    
    # Train the ensemble model
    ensemble.fit(X_train, y_train)
    
    # Test model
    y_pred = ensemble.predict(X_test)
    print('\nENSEMBLE MODEL PERFORMANCE')
    print('-' * 30)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return ensemble, scaler


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data


def main():
    print("Loading and preparing data...")
    data = get_clean_data()

    print("Training ensemble model...")
    model, scaler = create_model(data)

    print("\nSaving models...")
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Done! The ensemble model has been saved and is ready for use in Streamlit.")


if __name__ == '__main__':
    main()