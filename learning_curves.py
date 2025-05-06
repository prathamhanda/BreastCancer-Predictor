import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import seaborn as sns

def plot_learning_curves(estimator, X, y, title, cv=5):
    """
    Plot learning curves to analyze under/overfitting.
    
    Parameters:
    -----------
    estimator : sklearn estimator object
        The machine learning model
    X : array-like
        Training data
    y : array-like
        Target values
    title : str
        Title for the plot
    cv : int
        Number of cross-validation folds
    """
    # Set up the plot style
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Calculate learning curves
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=-1,
        scoring='neg_log_loss'
    )
    
    # Calculate mean and std
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.plot(train_sizes, train_mean, label='Training Loss', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, label='Validation Loss', color='red', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    # Customize plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Log Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    
    # Add analysis text
    gap = np.mean(val_mean - train_mean)
    final_train_score = train_mean[-1]
    final_val_score = val_mean[-1]
    
    analysis_text = f'Analysis:\n'
    if gap > 0.3:  # High gap between training and validation
        analysis_text += '- Potential overfitting\n'
    elif final_train_score > 0.5:  # High training loss
        analysis_text += '- Potential underfitting\n'
    else:
        analysis_text += '- Good fit\n'
    
    plt.text(0.02, 0.98, analysis_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'learning_curves_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_all_models_learning_curves(models_dict, X, y):
    """
    Plot learning curves for multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model name and model object pairs
    X : array-like
        Training data
    y : array-like
        Target values
    """
    for name, model in models_dict.items():
        plot_learning_curves(model, X, y, f'{name} Learning Curve') 