import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression  # 線性回歸
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from joblib import dump

# ---------- Load ----------
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze("columns")

X_test  = pd.read_csv("X_test.csv")
y_test  = pd.read_csv("y_test.csv").squeeze("columns")

print("Train shape:", X_train.shape)
print("Test  shape:", X_test.shape)

# ---------- 矯正 one-hot/bool 字串為數值 0/1（非前處理，純 dtype 修正） ----------
def coerce_boolish_inplace(df: pd.DataFrame, name: str) -> None:
    obj_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    if not obj_cols: 
        return
    mapping = {"1":1,"0":0,1:1,0:0,True:1,False:0,"yes":1,"no":0,"y":1,"n":0,"true":1,"false":0}
    for c in obj_cols:
        s = df[c]
        cn = pd.to_numeric(s, errors="coerce")
        if cn.notna().all():
            df[c] = cn.astype(np.float32); continue
        mapped = s.astype(str).str.strip().str.lower().map(mapping)
        if mapped.notna().all():
            df[c] = mapped.astype(np.float32); continue
        bad = sorted(set(s[mapped.isna()].unique().tolist()))
        raise TypeError(f"{name}.{c} has unmappable values: {bad[:10]}")

coerce_boolish_inplace(X_train, "X_train")
coerce_boolish_inplace(X_test,  "X_test")

# y -> numeric
y_train = pd.to_numeric(y_train, errors="raise").values.astype(np.float32)
y_test  = pd.to_numeric(y_test,  errors="raise").values.astype(np.float32)

# ---------- Sanity ----------
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train/Test columns mismatch (columns or order differ).")
if not all(np.issubdtype(dt, np.number) for dt in X_train.dtypes):
    bad = [c for c in X_train.columns if not np.issubdtype(X_train[c].dtype, np.number)]
    raise TypeError(f"Non-numeric columns remain: {bad}")

# Packed ndarray（float32 降記憶體）
X = np.asarray(X_train.values, dtype=np.float32, order="C")
Xt = np.asarray(X_test.values,  dtype=np.float32, order="C")

# ---------- Validation split ----------
X_tr, X_va, y_tr, y_va = train_test_split(X, y_train, test_size=0.2, random_state=0)

# ---------- Train Linear Regression Model ----------
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_tr, y_tr)

# Validate on validation set
y_va_pred = model.predict(X_va)
val_mape = mean_absolute_percentage_error(y_va, y_va_pred)
print(f"Validation MAPE: {val_mape*100:.3f}%")

# ---------- Retrain on full training set ----------
best_mdl = LinearRegression()
best_mdl.fit(X, y_train)

# ---------- Evaluate ----------
def report(y_true, y_pred, tag):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n{tag} Results:")
    print(f"MSE : {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"MAPE: {mape*100:.3f}%")

yhat_tr = best_mdl.predict(X)
yhat_te = best_mdl.predict(Xt)

print("\n" + "="*60)
print("Results (Linear Regression)")
print("="*60)

# Print regression formula (first 5 terms)
intercept = best_mdl.intercept_
coefficients = best_mdl.coef_
feature_names = X_train.columns.tolist()

print("\nRegression Formula (sorted by importance - absolute coefficient value):")
print(f"y = {intercept:.6f}")

# Sort coefficients by absolute value (importance)
coef_importance = [(abs(coefficients[i]), coefficients[i], feature_names[i]) for i in range(len(coefficients))]
coef_importance.sort(reverse=True, key=lambda x: x[0])

for abs_coef, coef, feature in coef_importance:
    sign = "+" if coef >= 0 else "-"
    print(f"    {sign} {abs_coef:.6f} * {feature}")

report(y_train, yhat_tr, "Train")
report(y_test,  yhat_te, "Test")

# ---------- Plot Results ----------
print("\nGenerating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Training Data - Predicted vs Actual
axes[0].scatter(y_train, yhat_tr, alpha=0.5, s=10)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Score', fontsize=12)
axes[0].set_ylabel('Predicted Score', fontsize=12)
axes[0].set_title('Training Set: Predicted vs Actual', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Plot 2: Test Data - Predicted vs Actual
axes[1].scatter(y_test, yhat_te, alpha=0.5, s=10, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Score', fontsize=12)
axes[1].set_ylabel('Predicted Score', fontsize=12)
axes[1].set_title('Test Set: Predicted vs Actual', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
print("Saved visualization to linear_regression_results.png")
plt.show()

# ---------- Save ----------
with open("predictions_linear.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["ID","y_pred"])
    for i, yhat in enumerate(yhat_te): w.writerow([i, yhat])
print("Saved predictions_linear.csv")

dump(best_mdl, "linear_regression_model.joblib")
print("Saved linear_regression_model.joblib")
