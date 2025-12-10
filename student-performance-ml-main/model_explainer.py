"""
Model Explainability using SHAP (SHapley Additive exPlanations)
Provides interpretability for all models and ensemble predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import load_data, ensure_dir, format_feature_name
from ensemble_model import StudentPerformanceEnsemble

class ModelExplainer:
    """
    Generate SHAP-based explanations for student performance models
    """
    
    def __init__(self, ensemble_model, X_train, X_test, feature_names=None):
        self.ensemble_model = ensemble_model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names or list(range(X_train.shape[1]))
        self.explainers = {}
        self.shap_values = {}
        
    def create_explainers(self):
        """Create SHAP explainers for each model"""
        print("\n" + "="*70)
        print("CREATING SHAP EXPLAINERS")
        print("="*70)
        
        # Use a background sample for faster computation
        background = shap.sample(self.X_train, min(100, len(self.X_train)))
        
        # Linear Regression - Linear explainer
        print("\n[1/4] Creating explainer for Linear Regression...")
        self.explainers['Linear Regression'] = shap.LinearExplainer(
            self.ensemble_model.models['Linear Regression'], 
            background
        )
        print("✓ Linear Regression explainer created")
        
        # Random Forest - Tree explainer
        print("\n[2/4] Creating explainer for Random Forest...")
        self.explainers['Random Forest'] = shap.TreeExplainer(
            self.ensemble_model.models['Random Forest']
        )
        print("✓ Random Forest explainer created")
        
        # XGBoost - Tree explainer
        print("\n[3/4] Creating explainer for XGBoost...")
        self.explainers['XGBoost'] = shap.TreeExplainer(
            self.ensemble_model.models['XGBoost']
        )
        print("✓ XGBoost explainer created")
        
        # Neural Network - Deep explainer (can be slow, so we'll use KernelExplainer)
        print("\n[4/4] Creating explainer for Neural Network...")
        print("   (Using Kernel explainer - this may take a moment...)")
        
        def nn_predict(X):
            return self.ensemble_model.models['Neural Network'].predict(X).flatten()
        
        self.explainers['Neural Network'] = shap.KernelExplainer(
            nn_predict, 
            background
        )
        print("✓ Neural Network explainer created")
        
    def calculate_shap_values(self, sample_size=100):
        """Calculate SHAP values for test set"""
        print("\n" + "="*70)
        print(f"CALCULATING SHAP VALUES (using {sample_size} samples)")
        print("="*70)
        
        # Use sample of test set for faster computation
        X_sample = self.X_test[:sample_size] if len(self.X_test) > sample_size else self.X_test
        
        for model_name, explainer in self.explainers.items():
            print(f"\nCalculating SHAP values for {model_name}...")
            
            try:
                if model_name == 'Neural Network':
                    # Kernel explainer returns explanations object
                    shap_vals = explainer.shap_values(X_sample, nsamples=100)
                else:
                    shap_vals = explainer.shap_values(X_sample)
                
                # Handle different SHAP value formats
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                    
                self.shap_values[model_name] = shap_vals
                print(f"✓ {model_name} SHAP values calculated")
                
            except Exception as e:
                print(f"⚠ Warning: Could not calculate SHAP for {model_name}: {str(e)}")
                
    def get_feature_importance(self, model_name):
        """Get feature importance from SHAP values"""
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not calculated for {model_name}")
        
        shap_vals = self.shap_values[model_name]
        importance = np.abs(shap_vals).mean(axis=0)
        
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return feature_importance
    
    def plot_feature_importance_comparison(self, top_n=10, save_path=None):
        """Compare feature importance across all models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(self.shap_values.keys()):
            ax = axes[idx]
            importance_df = self.get_feature_importance(model_name).head(top_n)
            
            # Create bar plot
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
            
            ax.set_xlabel('Mean |SHAP Value|', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}\nTop {top_n} Most Important Features', 
                        fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            # Format y-axis labels
            labels = [format_feature_name(f) for f in importance_df['Feature']]
            ax.set_yticklabels(labels, fontsize=9)
        
        plt.suptitle('Feature Importance Comparison Across Models (SHAP)', 
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved feature importance comparison to {save_path}")
        
        return fig
    
    def plot_shap_summary(self, model_name, save_path=None):
        """Create SHAP summary plot for a specific model"""
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not calculated for {model_name}")
        
        plt.figure(figsize=(10, 8))
        
        # Get sample data used for SHAP calculation
        sample_size = self.shap_values[model_name].shape[0]
        X_sample = self.X_test[:sample_size]
        
        shap.summary_plot(
            self.shap_values[model_name], 
            X_sample,
            feature_names=self.feature_names,
            show=False
        )
        
        plt.title(f'{model_name}: SHAP Summary Plot\n(Feature Impact on Predictions)', 
                 fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved SHAP summary plot to {save_path}")
        
        return plt.gcf()
    
    def plot_shap_waterfall(self, model_name, instance_idx=0, save_path=None):
        """Create SHAP waterfall plot for a single prediction"""
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not calculated for {model_name}")
        
        # Get the instance
        sample_size = self.shap_values[model_name].shape[0]
        if instance_idx >= sample_size:
            instance_idx = 0
        
        # Create explanation object
        if model_name == 'Linear Regression':
            explainer = self.explainers[model_name]
            expected_value = explainer.expected_value
        else:
            # For tree models, get expected value from explainer
            try:
                expected_value = self.explainers[model_name].expected_value
            except:
                expected_value = 0
        
        shap_vals = self.shap_values[model_name][instance_idx]
        
        # Create waterfall plot manually since shap.plots.waterfall needs Explanation object
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort features by absolute SHAP value
        indices = np.argsort(np.abs(shap_vals))[::-1][:10]  # Top 10
        
        y_pos = np.arange(len(indices))
        values = shap_vals[indices]
        features = [self.feature_names[i] for i in indices]
        
        colors = ['#ff0051' if v > 0 else '#008bfb' for v in values]
        
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([format_feature_name(f) for f in features])
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}: Feature Impact for Test Instance {instance_idx}\n' +
                    f'(Red = Increases Score, Blue = Decreases Score)', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved SHAP waterfall plot to {save_path}")
        
        return fig
    
    def explain_single_prediction(self, model_name, instance_idx=0):
        """Get textual explanation for a single prediction"""
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not calculated for {model_name}")
        
        sample_size = self.shap_values[model_name].shape[0]
        if instance_idx >= sample_size:
            instance_idx = 0
        
        shap_vals = self.shap_values[model_name][instance_idx]
        
        # Get top positive and negative contributors
        sorted_indices = np.argsort(shap_vals)
        
        top_positive_idx = sorted_indices[-3:][::-1]  # Top 3 positive
        top_negative_idx = sorted_indices[:3]  # Top 3 negative
        
        explanation = {
            'instance_idx': instance_idx,
            'top_positive_features': [
                {
                    'feature': self.feature_names[idx],
                    'shap_value': float(shap_vals[idx])
                }
                for idx in top_positive_idx
            ],
            'top_negative_features': [
                {
                    'feature': self.feature_names[idx],
                    'shap_value': float(shap_vals[idx])
                }
                for idx in top_negative_idx
            ]
        }
        
        return explanation


def main():
    """Generate all SHAP explanations and visualizations"""
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Load trained ensemble model
    print("\n" + "="*70)
    print("LOADING TRAINED ENSEMBLE MODEL")
    print("="*70)
    
    ensemble = StudentPerformanceEnsemble()
    ensemble.load_models()
    
    # Create explainer
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
    
    # Convert to numpy
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    
    explainer = ModelExplainer(ensemble, X_train_np, X_test_np, feature_names)
    
    # Create explainers
    explainer.create_explainers()
    
    # Calculate SHAP values
    explainer.calculate_shap_values(sample_size=200)
    
    # Generate visualizations
    ensure_dir('results')
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Feature importance comparison
    explainer.plot_feature_importance_comparison(
        top_n=10, 
        save_path='results/shap_feature_importance_comparison.png'
    )
    
    # Summary plots for each model
    for model_name in explainer.shap_values.keys():
        try:
            explainer.plot_shap_summary(
                model_name,
                save_path=f'results/shap_summary_{model_name.replace(" ", "_").lower()}.png'
            )
        except Exception as e:
            print(f"⚠ Could not create summary plot for {model_name}: {str(e)}")
    
    # Waterfall plots for sample predictions
    for model_name in list(explainer.shap_values.keys())[:2]:  # First 2 models
        try:
            explainer.plot_shap_waterfall(
                model_name,
                instance_idx=0,
                save_path=f'results/shap_waterfall_{model_name.replace(" ", "_").lower()}.png'
            )
        except Exception as e:
            print(f"⚠ Could not create waterfall plot for {model_name}: {str(e)}")
    
    # Print feature importance rankings
    print("\n" + "="*70)
    print("TOP 10 MOST IMPORTANT FEATURES (by model)")
    print("="*70)
    
    for model_name in explainer.shap_values.keys():
        print(f"\n{model_name}:")
        importance_df = explainer.get_feature_importance(model_name).head(10)
        for idx, row in importance_df.iterrows():
            print(f"  {row['Feature']:30s} | Importance: {row['Importance']:.4f}")
    
    # Save feature importance to CSV
    for model_name in explainer.shap_values.keys():
        importance_df = explainer.get_feature_importance(model_name)
        filename = f'results/feature_importance_{model_name.replace(" ", "_").lower()}.csv'
        importance_df.to_csv(filename, index=False)
        print(f"\n✓ Saved {model_name} feature importance to {filename}")
    
    print("\n" + "="*70)
    print("✓ SHAP ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
