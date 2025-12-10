"""
Weighted Ensemble Model for Student Performance Prediction
Combines Linear Regression, Random Forest, XGBoost, and Neural Network
with performance-based weights for optimal predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from scipy.optimize import minimize
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from utils import (
    load_data, calculate_metrics, print_metrics, 
    plot_predictions, ensure_dir, save_json
)

class StudentPerformanceEnsemble:
    """
    Ensemble model combining multiple ML algorithms for student performance prediction.
    Uses weighted averaging based on validation performance.
    """
    
    def __init__(self, models_dir='models', results_dir='results'):
        self.models_dir = models_dir
        self.results_dir = results_dir
        ensure_dir(models_dir)
        ensure_dir(results_dir)
        
        self.models = {}
        self.weights = {}
        self.feature_names = None
        self.trained = False
        
    def build_linear_regression(self):
        """Build Linear Regression model"""
        return LinearRegression()
    
    def build_random_forest(self):
        """Build Random Forest model with optimized hyperparameters"""
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features=None,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    def build_xgboost(self):
        """Build XGBoost model with optimized hyperparameters"""
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    def build_neural_network(self, input_dim):
        """Build Neural Network model"""
        model = Sequential([
            Dense(256, activation="relu", kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.001), input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation="relu", kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation="relu", kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation="relu", kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, validation_split=0.2):
        """
        Train all models and calculate ensemble weights
        
        Args:
            X_train: Training features
            y_train: Training targets
            validation_split: Fraction of training data to use for validation
        """
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE MODEL")
        print("="*70)
        
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        # Convert to numpy if pandas
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # Create validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        validation_predictions = {}
        
        # 1. Train Linear Regression
        print("\n[1/4] Training Linear Regression...")
        lr_model = self.build_linear_regression()
        lr_model.fit(X_tr, y_tr)
        self.models['Linear Regression'] = lr_model
        validation_predictions['Linear Regression'] = lr_model.predict(X_val)
        print("[OK] Linear Regression trained")
        
        # 2. Train Random Forest
        print("\n[2/4] Training Random Forest...")
        rf_model = self.build_random_forest()
        rf_model.fit(X_tr, y_tr)
        self.models['Random Forest'] = rf_model
        validation_predictions['Random Forest'] = rf_model.predict(X_val)
        print("[OK] Random Forest trained")
        
        # 3. Train XGBoost
        print("\n[3/4] Training XGBoost...")
        xgb_model = self.build_xgboost()
        xgb_model.fit(X_tr, y_tr)
        self.models['XGBoost'] = xgb_model
        validation_predictions['XGBoost'] = xgb_model.predict(X_val)
        print("[OK] XGBoost trained")
        
        # 4. Train Neural Network
        print("\n[4/4] Training Neural Network...")
        nn_model = self.build_neural_network(X_tr.shape[1])
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        lr_schedule = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
        
        nn_model.fit(
            X_tr, y_tr,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=[early_stop, lr_schedule],
            verbose=0
        )
        
        self.models['Neural Network'] = nn_model
        validation_predictions['Neural Network'] = nn_model.predict(X_val).flatten()
        print("[OK] Neural Network trained")
        
        # Calculate weights based on validation performance
        print("\n" + "="*70)
        print("CALCULATING ENSEMBLE WEIGHTS")
        print("="*70)
        
        val_metrics = {}
        for model_name, preds in validation_predictions.items():
            rmse = np.sqrt(np.mean((y_val - preds) ** 2))
            val_metrics[model_name] = rmse
        
        print("\nValidation RMSE for each model:")
        for model_name, rmse in val_metrics.items():
            print(f"  {model_name:20s} | Val RMSE: {rmse:.4f}")
        
        # Use optimization to find best weights
        print("\n🔬 Using scipy.optimize to find mathematically optimal weights...")
        self.weights = self.optimize_weights(validation_predictions, y_val)
        
        print("\n✓ Optimized Ensemble Weights:")
        for model_name, weight in self.weights.items():
            print(f"  {model_name:20s} | Weight: {weight:.4f}")
        
        # Validate ensemble performance
        ensemble_pred_val = sum(
            validation_predictions[model_name] * self.weights[model_name]
            for model_name in self.models.keys()
        )
        ensemble_rmse = np.sqrt(np.mean((y_val - ensemble_pred_val) ** 2))
        print(f"\n✅ Ensemble Validation RMSE: {ensemble_rmse:.4f}")
        print(f"   Best Single Model RMSE: {min(val_metrics.values()):.4f}")
        
        if ensemble_rmse <= min(val_metrics.values()):
            improvement = ((min(val_metrics.values()) - ensemble_rmse) / min(val_metrics.values())) * 100
            print(f"   🎯 Ensemble improved by {improvement:.2f}%")
        else:
            print(f"   ⚠️  Note: Ensemble RMSE is close to best model (optimization constraint)")
        
        self.trained = True
        print("\n✓ Ensemble model training complete!")
        
        return val_metrics
    
    def optimize_weights(self, predictions_dict, y_true):
        """
        Find optimal ensemble weights using scipy optimization
        
        Args:
            predictions_dict: Dictionary of {model_name: predictions}
            y_true: True target values
            
        Returns:
            Dictionary of optimized weights
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        # Convert predictions to matrix (rows=samples, cols=models)
        pred_matrix = np.column_stack([predictions_dict[name] for name in model_names])
        
        # Objective function: RMSE of weighted ensemble
        def objective(weights):
            ensemble_pred = pred_matrix @ weights  # Matrix multiplication
            rmse = np.sqrt(np.mean((y_true - ensemble_pred) ** 2))
            return rmse
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: each weight between 0 and 1
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_models) / n_models
        
        # Optimize using Sequential Least Squares Programming (SLSQP)
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"⚠️  Optimization warning: {result.message}")
            print("   Falling back to equal weights")
            optimized_weights = initial_weights
        else:
            optimized_weights = result.x
        
        # Convert to dictionary
        weights_dict = {model_names[i]: optimized_weights[i] for i in range(n_models)}
        
        return weights_dict
    
    def predict(self, X):
        """
        Make predictions using weighted ensemble
        
        Args:
            X: Features to predict on
            
        Returns:
            Weighted ensemble predictions
        """
        if not self.trained:
            raise ValueError("Model not trained! Call train() first.")
        
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get predictions from all models
        predictions = {}
        predictions['Linear Regression'] = self.models['Linear Regression'].predict(X)
        predictions['Random Forest'] = self.models['Random Forest'].predict(X)
        predictions['XGBoost'] = self.models['XGBoost'].predict(X)
        predictions['Neural Network'] = self.models['Neural Network'].predict(X).flatten()
        
        # Calculate weighted average
        ensemble_pred = sum(
            predictions[model_name] * self.weights[model_name]
            for model_name in self.models.keys()
        )
        
        return ensemble_pred
    
    def predict_all(self, X):
        """
        Get predictions from all individual models plus ensemble
        
        Returns:
            Dictionary of {model_name: predictions}
        """
        if not self.trained:
            raise ValueError("Model not trained! Call train() first.")
        
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = {}
        predictions['Linear Regression'] = self.models['Linear Regression'].predict(X)
        predictions['Random Forest'] = self.models['Random Forest'].predict(X)
        predictions['XGBoost'] = self.models['XGBoost'].predict(X)
        predictions['Neural Network'] = self.models['Neural Network'].predict(X).flatten()
        predictions['Ensemble'] = self.predict(X)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate all models on test set
        
        Returns:
            DataFrame with metrics for all models
        """
        print("\n" + "="*70)
        print("EVALUATING ENSEMBLE ON TEST SET")
        print("="*70)
        
        # Get predictions from all models
        all_predictions = self.predict_all(X_test)
        
        # Calculate metrics for each model
        results = []
        for model_name, preds in all_predictions.items():
            metrics = calculate_metrics(y_test, preds, model_name)
            results.append(metrics)
            print_metrics(metrics, "Test")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('RMSE')
        
        return results_df
    
    def save_models(self):
        """Save all trained models and weights"""
        if not self.trained:
            raise ValueError("No trained models to save!")
        
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        # Save sklearn models
        joblib.dump(self.models['Linear Regression'], 
                   f'{self.models_dir}/linear_regression.joblib')
        print("✓ Saved Linear Regression")
        
        joblib.dump(self.models['Random Forest'], 
                   f'{self.models_dir}/random_forest.joblib')
        print("✓ Saved Random Forest")
        
        joblib.dump(self.models['XGBoost'], 
                   f'{self.models_dir}/xgboost.joblib')
        print("✓ Saved XGBoost")
        
        # Save Keras model
        self.models['Neural Network'].save(f'{self.models_dir}/neural_network.keras')
        print("✓ Saved Neural Network")
        
        # Save weights and metadata
        metadata = {
            'weights': self.weights,
            'feature_names': self.feature_names
        }
        save_json(metadata, f'{self.models_dir}/ensemble_config.json')
        print("✓ Saved ensemble configuration")
        
    def load_models(self):
        """Load previously saved models and weights"""
        print("\n" + "="*70)
        print("LOADING MODELS")
        print("="*70)
        
        self.models['Linear Regression'] = joblib.load(
            f'{self.models_dir}/linear_regression.joblib')
        print("✓ Loaded Linear Regression")
        
        self.models['Random Forest'] = joblib.load(
            f'{self.models_dir}/random_forest.joblib')
        print("✓ Loaded Random Forest")
        
        self.models['XGBoost'] = joblib.load(
            f'{self.models_dir}/xgboost.joblib')
        print("✓ Loaded XGBoost")
        
        self.models['Neural Network'] = load_model(
            f'{self.models_dir}/neural_network.keras')
        print("✓ Loaded Neural Network")
        
        # Load weights and metadata
        with open(f'{self.models_dir}/ensemble_config.json', 'r') as f:
            metadata = json.load(f)
        self.weights = metadata['weights']
        self.feature_names = metadata['feature_names']
        print("✓ Loaded ensemble configuration")
        
        self.trained = True


def main():
    """Main training and evaluation script"""
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Create and train ensemble
    ensemble = StudentPerformanceEnsemble()
    val_metrics = ensemble.train(X_train, y_train)
    
    # Evaluate on test set
    test_results = ensemble.evaluate(X_test, y_test)
    
    # Save results
    test_results.to_csv('results/ensemble_comparison.csv', index=False)
    print(f"\n✓ Saved results to results/ensemble_comparison.csv")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: RMSE Comparison
    ax = axes[0, 0]
    models_sorted = test_results.sort_values('RMSE')
    colors = ['#2ecc71' if 'Ensemble' in m else '#3498db' 
              for m in models_sorted['Model']]
    ax.barh(models_sorted['Model'], models_sorted['RMSE'], color=colors)
    ax.set_xlabel('RMSE (Lower is Better)', fontsize=11, fontweight='bold')
    ax.set_title('Test Set: RMSE Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: R² Comparison
    ax = axes[0, 1]
    models_r2 = test_results.sort_values('R²', ascending=False)
    colors = ['#2ecc71' if 'Ensemble' in m else '#3498db' 
              for m in models_r2['Model']]
    ax.barh(models_r2['Model'], models_r2['R²'], color=colors)
    ax.set_xlabel('R² Score (Higher is Better)', fontsize=11, fontweight='bold')
    ax.set_title('Test Set: R² Score Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 3: MAE Comparison
    ax = axes[1, 0]
    models_mae = test_results.sort_values('MAE')
    colors = ['#2ecc71' if 'Ensemble' in m else '#3498db' 
              for m in models_mae['Model']]
    ax.barh(models_mae['Model'], models_mae['MAE'], color=colors)
    ax.set_xlabel('MAE (Lower is Better)', fontsize=11, fontweight='bold')
    ax.set_title('Test Set: MAE Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 4: Metrics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    table_data = test_results[['Model', 'RMSE', 'MAE', 'R²']].round(4)
    table = ax.table(cellText=table_data.values, 
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Highlight ensemble row in green
    for i, model in enumerate(table_data['Model']):
        if 'Ensemble' in model:
            for j in range(len(table_data.columns)):
                table[(i+1, j)].set_facecolor('#90EE90')
    
    plt.suptitle('Student Performance Prediction: Ensemble Model Performance', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('results/ensemble_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to results/ensemble_performance.png")
    
    # Save models
    ensemble.save_models()
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 FINAL SUMMARY")
    print("="*70)
    best_model = test_results.iloc[0]
    print(f"\nBest Model: {best_model['Model']}")
    print(f"Test RMSE: {best_model['RMSE']:.4f}")
    print(f"Test MAE:  {best_model['MAE']:.4f}")
    print(f"Test R²:   {best_model['R²']:.4f}")
    
    if 'Ensemble' in best_model['Model']:
        print("\n✅ Ensemble model outperforms all individual models!")
    else:
        ensemble_metrics = test_results[test_results['Model'] == 'Ensemble'].iloc[0]
        diff = ((ensemble_metrics['RMSE'] - best_model['RMSE']) / best_model['RMSE']) * 100
        print(f"\n⚠ Best individual model outperforms ensemble by {abs(diff):.2f}%")
        print(f"   Consider using {best_model['Model']} for final predictions")
    
    print("\n" + "="*70)
    

if __name__ == "__main__":
    main()
