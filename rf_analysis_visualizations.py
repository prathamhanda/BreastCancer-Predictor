import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
import time
import os

# Create directory for visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Load and prepare data
print("Loading data...")
data = pd.read_csv("data/data.csv")
data = data.drop(['Unnamed: 32', 'id'], axis=1)
data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

# Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def plot_tree_depth_analysis():
    """Plot the impact of tree depth on model performance"""
    depths = [3, 5, 7, 10, 15, 20, 25, 30, None]
    train_scores = []
    test_scores = []
    fit_times = []

    for depth in depths:
        rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        start_time = time.time()
        rf.fit(X_train_scaled, y_train)
        fit_times.append(time.time() - start_time)
        train_scores.append(rf.score(X_train_scaled, y_train))
        test_scores.append(rf.score(X_test_scaled, y_test))

    plt.figure(figsize=(15, 5))
    
    # Plot 1: Accuracy vs Tree Depth
    plt.subplot(1, 2, 1)
    plt.plot(range(len(depths)), train_scores, 'o-', label='Training Accuracy')
    plt.plot(range(len(depths)), test_scores, 'o-', label='Testing Accuracy')
    plt.xticks(range(len(depths)), [str(d) if d is not None else 'None' for d in depths])
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Impact of Tree Depth on Model Performance')
    plt.legend()
    plt.grid(True)

    # Plot 2: Training Time vs Tree Depth
    plt.subplot(1, 2, 2)
    plt.plot(range(len(depths)), fit_times, 'o-', color='green')
    plt.xticks(range(len(depths)), [str(d) if d is not None else 'None' for d in depths])
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Tree Depth')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/tree_depth_analysis.png')
    plt.close()

def plot_ensemble_size_analysis():
    """Plot the impact of ensemble size on model performance"""
    n_estimators_range = [10, 50, 100, 200, 300, 500]
    train_scores = []
    test_scores = []
    fit_times = []

    for n_estimators in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=15, random_state=42)
        start_time = time.time()
        rf.fit(X_train_scaled, y_train)
        fit_times.append(time.time() - start_time)
        train_scores.append(rf.score(X_train_scaled, y_train))
        test_scores.append(rf.score(X_test_scaled, y_test))

    plt.figure(figsize=(15, 5))
    
    # Plot 1: Accuracy vs Ensemble Size
    plt.subplot(1, 2, 1)
    plt.plot(n_estimators_range, train_scores, 'o-', label='Training Accuracy')
    plt.plot(n_estimators_range, test_scores, 'o-', label='Testing Accuracy')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Impact of Ensemble Size on Model Performance')
    plt.legend()
    plt.grid(True)

    # Plot 2: Training Time vs Ensemble Size
    plt.subplot(1, 2, 2)
    plt.plot(n_estimators_range, fit_times, 'o-', color='green')
    plt.xlabel('Number of Trees')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Ensemble Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/ensemble_size_analysis.png')
    plt.close()

def plot_feature_importance():
    """Plot feature importance analysis"""
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Get feature importance
    importance = rf.feature_importances_
    features = X.columns
    
    # Sort features by importance
    feat_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_importance = feat_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_importance.head(15))
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()

def plot_min_samples_analysis():
    """Plot the impact of min_samples_split and min_samples_leaf"""
    min_samples_splits = [2, 5, 10, 20, 50]
    min_samples_leafs = [1, 2, 4, 8, 16]
    
    results = np.zeros((len(min_samples_splits), len(min_samples_leafs)))
    
    for i, split in enumerate(min_samples_splits):
        for j, leaf in enumerate(min_samples_leafs):
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=split,
                min_samples_leaf=leaf,
                random_state=42
            )
            rf.fit(X_train_scaled, y_train)
            results[i, j] = rf.score(X_test_scaled, y_test)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, 
                xticklabels=min_samples_leafs,
                yticklabels=min_samples_splits,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd')
    plt.xlabel('min_samples_leaf')
    plt.ylabel('min_samples_split')
    plt.title('Model Performance for Different Tree Splitting Parameters')
    plt.tight_layout()
    plt.savefig('visualizations/min_samples_analysis.png')
    plt.close()

def plot_learning_curves():
    """Plot learning curves for different tree depths"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    depths = [5, 15, None]
    
    plt.figure(figsize=(15, 5))
    
    for idx, depth in enumerate(depths):
        rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            rf, X_train_scaled, y_train,
            train_sizes=train_sizes,
            cv=5,
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.subplot(1, 3, idx+1)
        plt.plot(train_sizes_abs, train_mean, 'o-', label='Training Score')
        plt.plot(train_sizes_abs, test_mean, 'o-', label='Cross-validation Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title(f'Learning Curves (max_depth={depth})')
        plt.legend(loc='best')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/learning_curves.png')
    plt.close()

if __name__ == "__main__":
    print("Generating visualizations...")
    plot_tree_depth_analysis()
    print("1/5: Tree depth analysis completed")
    plot_ensemble_size_analysis()
    print("2/5: Ensemble size analysis completed")
    plot_feature_importance()
    print("3/5: Feature importance analysis completed")
    plot_min_samples_analysis()
    print("4/5: Min samples analysis completed")
    plot_learning_curves()
    print("5/5: Learning curves analysis completed")
    print("All visualizations have been generated in the 'visualizations' directory") 