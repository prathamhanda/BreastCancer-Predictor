from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm(X_train, X_test, y_train, y_test):
    # Define parameter grid optimized for medical diagnosis
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'class_weight': ['balanced', None],
        'probability': [True],  # Required for probability estimates
        'cache_size': [2000]  # Increase cache size for faster training
    }
    
    # Initialize model with settings suitable for medical data
    svm = SVC(
        random_state=42,
        verbose=0,
        max_iter=2000  # Increase max iterations for better convergence
    )
    
    # Perform grid search with stratified k-fold
    grid_search = GridSearchCV(
        svm, 
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
    
    return best_model, accuracy, grid_search.best_params_ 