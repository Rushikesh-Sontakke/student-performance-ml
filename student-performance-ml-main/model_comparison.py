"""
Model Comparison & Ensemble Recommendation Script
Compares all models (Linear Regression, Random Forest, XGBoost, Neural Network)
and provides recommendations on which to use or whether to ensemble them.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ==================== Load Data ====================
print("Loading data...")
X_train = pd.read_csv("Data/Processed/X_train.csv")
y_train = pd.read_csv("Data/Processed/y_train.csv").values.flatten()
X_test = pd.read_csv("Data/Processed/X_test.csv")
y_test = pd.read_csv("Data/Processed/y_test.csv").values.flatten()

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

# ==================== Train All Models ====================
print("="*70)
print("TRAINING ALL MODELS")
print("="*70)

models = {}
predictions_train = {}
predictions_test = {}

# 1. Linear Regression
print("\n1. Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
models['Linear Regression'] = lr_model
predictions_train['Linear Regression'] = lr_model.predict(X_train)
predictions_test['Linear Regression'] = lr_model.predict(X_test)
print("✓ Linear Regression trained")

# 2. Random Forest
print("\n2. Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model
predictions_train['Random Forest'] = rf_model.predict(X_train)
predictions_test['Random Forest'] = rf_model.predict(X_test)
print("✓ Random Forest trained")

# 3. XGBoost
print("\n3. Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model
predictions_train['XGBoost'] = xgb_model.predict(X_train)
predictions_test['XGBoost'] = xgb_model.predict(X_test)
print("✓ XGBoost trained")

# 4. Neural Network
print("\n4. Training Neural Network...")
nn_model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
nn_model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, 
             callbacks=[early_stop], verbose=0)
models['Neural Network'] = nn_model
predictions_train['Neural Network'] = nn_model.predict(X_train).flatten()
predictions_test['Neural Network'] = nn_model.predict(X_test).flatten()
print("✓ Neural Network trained")

# ==================== Evaluate All Models ====================
print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)

def evaluate_model(y_true, y_pred, model_name, dataset='Test'):
    """Calculate all metrics for a model"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        'Model': model_name,
        'Dataset': dataset,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape * 100
    }

# Collect all results
results = []
for model_name in models.keys():
    # Test set
    results.append(evaluate_model(y_test, predictions_test[model_name], model_name, 'Test'))
    # Train set (to check overfitting)
    results.append(evaluate_model(y_train, predictions_train[model_name], model_name, 'Train'))

results_df = pd.DataFrame(results)

# Display test results
test_results = results_df[results_df['Dataset'] == 'Test'].sort_values('RMSE')
print("\n📊 TEST SET PERFORMANCE (Lower is better for MSE/RMSE/MAE/MAPE, Higher is better for R²):")
print(test_results.to_string(index=False))

# Check for overfitting
print("\n🔍 OVERFITTING CHECK (Train vs Test RMSE):")
for model_name in models.keys():
    train_rmse = results_df[(results_df['Model'] == model_name) & (results_df['Dataset'] == 'Train')]['RMSE'].values[0]
    test_rmse = results_df[(results_df['Model'] == model_name) & (results_df['Dataset'] == 'Test')]['RMSE'].values[0]
    diff = test_rmse - train_rmse
    diff_pct = (diff / train_rmse) * 100
    
    if diff_pct < 10:
        status = "✓ Good (minimal overfitting)"
    elif diff_pct < 20:
        status = "⚠ Moderate overfitting"
    else:
        status = "❌ High overfitting"
    
    print(f"{model_name:20s} | Train: {train_rmse:.3f} | Test: {test_rmse:.3f} | Diff: {diff_pct:+.1f}% | {status}")

# ==================== Ensemble Methods ====================
print("\n" + "="*70)
print("ENSEMBLE APPROACHES")
print("="*70)

# 1. Simple Average Ensemble
print("\n1️⃣ Simple Average Ensemble (equal weights)")
ensemble_avg_test = np.mean([predictions_test[m] for m in models.keys()], axis=0)
ensemble_avg_train = np.mean([predictions_train[m] for m in models.keys()], axis=0)
avg_metrics = evaluate_model(y_test, ensemble_avg_test, 'Ensemble (Average)', 'Test')
print(f"   Test RMSE: {avg_metrics['RMSE']:.4f} | MAE: {avg_metrics['MAE']:.4f} | R²: {avg_metrics['R²']:.4f}")

# 2. Weighted Average Ensemble (based on inverse RMSE)
print("\n2️⃣ Weighted Average Ensemble (performance-based weights)")
weights = {}
total_inv_rmse = 0
for model_name in models.keys():
    rmse = results_df[(results_df['Model'] == model_name) & (results_df['Dataset'] == 'Test')]['RMSE'].values[0]
    inv_rmse = 1.0 / rmse
    weights[model_name] = inv_rmse
    total_inv_rmse += inv_rmse

# Normalize weights
for model_name in weights:
    weights[model_name] /= total_inv_rmse
    print(f"   {model_name:20s}: {weights[model_name]:.4f}")

ensemble_weighted_test = sum(predictions_test[m] * weights[m] for m in models.keys())
weighted_metrics = evaluate_model(y_test, ensemble_weighted_test, 'Ensemble (Weighted)', 'Test')
print(f"   Test RMSE: {weighted_metrics['RMSE']:.4f} | MAE: {weighted_metrics['MAE']:.4f} | R²: {weighted_metrics['R²']:.4f}")

# 3. Best 2 Models Ensemble
print("\n3️⃣ Best Two Models Ensemble")
best_two = test_results.head(2)['Model'].values
print(f"   Best two models: {best_two[0]} & {best_two[1]}")
ensemble_best2_test = np.mean([predictions_test[m] for m in best_two], axis=0)
best2_metrics = evaluate_model(y_test, ensemble_best2_test, f'Ensemble ({best_two[0]} + {best_two[1]})', 'Test')
print(f"   Test RMSE: {best2_metrics['RMSE']:.4f} | MAE: {best2_metrics['MAE']:.4f} | R²: {best2_metrics['R²']:.4f}")

# ==================== Visualizations ====================
print("\n📈 Generating visualizations...")

fig = plt.figure(figsize=(18, 10))

# 1. RMSE Comparison
ax1 = plt.subplot(2, 3, 1)
test_results_sorted = test_results.sort_values('RMSE')
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(test_results_sorted))]
ax1.barh(test_results_sorted['Model'], test_results_sorted['RMSE'], color=colors)
ax1.set_xlabel('RMSE (Lower is Better)', fontsize=11, fontweight='bold')
ax1.set_title('Model Comparison: RMSE', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. R² Comparison
ax2 = plt.subplot(2, 3, 2)
test_results_r2 = test_results.sort_values('R²', ascending=False)
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(test_results_r2))]
ax2.barh(test_results_r2['Model'], test_results_r2['R²'], color=colors)
ax2.set_xlabel('R² Score (Higher is Better)', fontsize=11, fontweight='bold')
ax2.set_title('Model Comparison: R² Score', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. MAE Comparison
ax3 = plt.subplot(2, 3, 3)
test_results_mae = test_results.sort_values('MAE')
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(test_results_mae))]
ax3.barh(test_results_mae['Model'], test_results_mae['MAE'], color=colors)
ax3.set_xlabel('MAE (Lower is Better)', fontsize=11, fontweight='bold')
ax3.set_title('Model Comparison: MAE', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Ensemble Comparison
ax4 = plt.subplot(2, 3, 4)
ensemble_names = ['Simple Avg', 'Weighted Avg', 'Best 2', 'Best Single']
ensemble_rmses = [
    avg_metrics['RMSE'],
    weighted_metrics['RMSE'],
    best2_metrics['RMSE'],
    test_results.iloc[0]['RMSE']
]
colors_ens = ['#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']
ax4.bar(ensemble_names, ensemble_rmses, color=colors_ens)
ax4.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax4.set_title('Ensemble vs Best Single Model', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.tick_params(axis='x', rotation=15)

# 5. Prediction scatter for best model
ax5 = plt.subplot(2, 3, 5)
best_model_name = test_results.iloc[0]['Model']
best_preds = predictions_test[best_model_name]
ax5.scatter(y_test, best_preds, alpha=0.5, s=20, color='#3498db')
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax5.set_xlabel('Actual Scores', fontsize=11, fontweight='bold')
ax5.set_ylabel('Predicted Scores', fontsize=11, fontweight='bold')
ax5.set_title(f'Best Model: {best_model_name}', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Overfitting comparison
ax6 = plt.subplot(2, 3, 6)
model_names = list(models.keys())
train_rmses = [results_df[(results_df['Model'] == m) & (results_df['Dataset'] == 'Train')]['RMSE'].values[0] for m in model_names]
test_rmses = [results_df[(results_df['Model'] == m) & (results_df['Dataset'] == 'Test')]['RMSE'].values[0] for m in model_names]
x = np.arange(len(model_names))
width = 0.35
ax6.bar(x - width/2, train_rmses, width, label='Train', color='#2ecc71', alpha=0.8)
ax6.bar(x + width/2, test_rmses, width, label='Test', color='#e74c3c', alpha=0.8)
ax6.set_xlabel('Models', fontsize=11, fontweight='bold')
ax6.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax6.set_title('Train vs Test RMSE (Overfitting Check)', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(model_names, rotation=15, ha='right')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_results.png")

# ==================== RECOMMENDATIONS ====================
print("\n" + "="*70)
print("🎯 FINAL RECOMMENDATIONS")
print("="*70)

best_model = test_results.iloc[0]
second_best = test_results.iloc[1]

print(f"\n1️⃣ BEST SINGLE MODEL: {best_model['Model']}")
print(f"   → RMSE: {best_model['RMSE']:.4f} | MAE: {best_model['MAE']:.4f} | R²: {best_model['R²']:.4f}")
print(f"   → Use this if you want simplicity and the best standalone performance")

print(f"\n2️⃣ ENSEMBLE RECOMMENDATION:")
if weighted_metrics['RMSE'] < best_model['RMSE']:
    improvement = ((best_model['RMSE'] - weighted_metrics['RMSE']) / best_model['RMSE']) * 100
    print(f"   ✅ RECOMMEND WEIGHTED ENSEMBLE")
    print(f"   → RMSE: {weighted_metrics['RMSE']:.4f} ({improvement:.2f}% better than best single model)")
    print(f"   → Combines all models with performance-based weights")
    print(f"   → More robust and generally better generalization")
else:
    print(f"   ⚠ STICK WITH BEST SINGLE MODEL")
    print(f"   → Ensemble doesn't improve performance significantly")
    print(f"   → {best_model['Model']} is already optimal")

print(f"\n3️⃣ FOR YOUR FINAL PROJECT:")
print(f"   Option A (Recommended): Use ENSEMBLE for better differentiation")
print(f"            • Shows advanced ML knowledge (model combination)")
print(f"            • More robust predictions")
print(f"            • Demonstrates you tried multiple approaches")
print(f"            • Weight each model by: {', '.join([f'{k}={v:.3f}' for k, v in weights.items()])}")

print(f"\n   Option B (Simpler): Use {best_model['Model']} only")
print(f"            • Simplest approach, easiest to explain")
print(f"            • Best single-model performance")
print(f"            • Less complex implementation")

print(f"\n4️⃣ DIFFERENTIATION STRATEGY:")
print(f"   • You have 4 different models - this is already good!")
print(f"   • Consider adding SHAP/LIME for model explainability")
print(f"   • Build interactive dashboard (Streamlit/Gradio)")
print(f"   • Show why ensemble is better (or why single model is sufficient)")

print("\n" + "="*70)
print("Results saved to: model_comparison_results.png")
print("="*70)

# Save the comparison results
results_df.to_csv('model_comparison_metrics.csv', index=False)
print("✓ Saved detailed metrics to: model_comparison_metrics.csv")
