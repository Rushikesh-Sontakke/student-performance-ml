"""
Utility Functions for Student Performance ML Project
Shared utilities for data loading, metrics, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import json
import os

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(base_path="Data/Processed"):
    """Load preprocessed train and test data"""
    X_train = pd.read_csv(f"{base_path}/X_train.csv")
    y_train = pd.read_csv(f"{base_path}/y_train.csv").values.flatten()
    X_test = pd.read_csv(f"{base_path}/X_test.csv")
    y_test = pd.read_csv(f"{base_path}/y_test.csv").values.flatten()
    
    # Convert boolean columns (stored as strings "True"/"False") to numeric (1/0)
    # This is necessary for TensorFlow/Keras which cannot handle object dtype
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].dtype == 'bool':
            X_train[col] = X_train[col].astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0}).astype(float)
            X_test[col] = X_test[col].astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0}).astype(float)
    
    print(f"[OK] Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calculate comprehensive regression metrics"""
    metrics = {
        'Model': model_name,
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }
    return metrics


def print_metrics(metrics, dataset="Test"):
    """Pretty print model metrics"""
    print(f"\n{'='*60}")
    print(f"{metrics['Model']} - {dataset} Set Performance")
    print(f"{'='*60}")
    print(f"MSE  : {metrics['MSE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"R²   : {metrics['R²']:.4f}")
    print(f"MAPE : {metrics['MAPE']:.2f}%")
    print(f"{'='*60}")


def compare_models(results_dict, y_test, metric='RMSE'):
    """
    Compare multiple models and return sorted DataFrame
    
    Args:
        results_dict: Dict of {model_name: predictions}
        y_test: True values
        metric: Metric to sort by (default: RMSE)
    """
    comparison = []
    for model_name, predictions in results_dict.items():
        metrics = calculate_metrics(y_test, predictions, model_name)
        comparison.append(metrics)
    
    df = pd.DataFrame(comparison)
    df = df.sort_values(metric, ascending=(metric != 'R²'))
    return df


def plot_predictions(y_true, y_pred, model_name, save_path=None):
    """Plot actual vs predicted values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Exam Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Exam Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    return fig


def plot_residuals(y_true, y_pred, model_name, save_path=None):
    """Plot residuals distribution"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name} - Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved residual plot to {save_path}")
    return fig


def save_json(data, filepath):
    """Save dictionary to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved to {filepath}")


def load_json(filepath):
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_comparison_table(metrics_df, save_path=None):
    """Create and optionally save a styled comparison table"""
    # Sort by RMSE (lower is better)
    df_sorted = metrics_df.sort_values('RMSE').reset_index(drop=True)
    
    # Highlight best values
    def highlight_best(s):
        if s.name in ['MSE', 'RMSE', 'MAE', 'MAPE']:
            is_best = s == s.min()
        elif s.name == 'R²':
            is_best = s == s.max()
        else:
            return [''] * len(s)
        return ['background-color: lightgreen' if v else '' for v in is_best]
    
    styled = df_sorted.style.apply(highlight_best, subset=['MSE', 'RMSE', 'MAE', 'R²', 'MAPE'])
    
    if save_path:
        styled.to_html(save_path)
        print(f"✓ Saved comparison table to {save_path}")
    
    return styled


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ Created directory: {directory}")


def format_feature_name(feature_name):
    """Convert feature name to readable format"""
    # Convert snake_case or PascalCase to Title Case with spaces
    name = feature_name.replace('_', ' ')
    return ' '.join(word.capitalize() for word in name.split())
