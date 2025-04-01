from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, X_test, y_train, y_test):
    # Define parameter grid optimized for medical diagnosis
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Initialize model with settings suitable for medical data
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        oob_score=True,  # Out-of-bag score for validation
        verbose=0
    )
    
    # Perform grid search with stratified k-fold
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        refit='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Get accuracy
    accuracy = best_model.score(X_test, y_test)
    
    # Get feature importance
    feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
    
    return best_model, accuracy, grid_search.best_params_, feature_importance 