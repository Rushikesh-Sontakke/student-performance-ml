import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

# --- 1. 資料讀取 (注意路徑) ---
# 隊友的 code 寫 "X_train.csv"，但看你的截圖資料好像在 Data 資料夾
# 這裡假設你的 script 在 src/ 執行，而資料在 ../Data/Processed/ (請依實際情況調整)
# 如果檔案就在同目錄，改回 "X_train.csv" 即可
DATA_PATH = "../Data/Processed/"  # 或是 "" 如果檔案在同層
if not os.path.exists(DATA_PATH + "X_train.csv"):
    print(f"⚠️ 找不到資料，請確認路徑。嘗試讀取當前目錄...")
    DATA_PATH = ""

print("Loading data...")
X_train = pd.read_csv(f"{DATA_PATH}X_train.csv")
y_train = pd.read_csv(f"{DATA_PATH}y_train.csv").values.flatten()
X_test = pd.read_csv(f"{DATA_PATH}X_test.csv")
y_test = pd.read_csv(f"{DATA_PATH}y_test.csv").values.flatten()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# --- 2. 建立與訓練 Random Forest 模型 ---
print("\nTraining Random Forest Model...")
# n_estimators: 樹的數量 (越多通常越穩，但也越慢)
# max_depth: 樹的深度 (太深會過擬合，None 代表不限)
# n_jobs=-1: 用電腦所有的 CPU 核心去跑，會快很多
model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=None,       
    random_state=42,
    n_jobs=-1             
)

model.fit(X_train, y_train)

# --- 3. 儲存模型 ---
# Scikit-learn 模型通常用 joblib 儲存
joblib.dump(model, "exam_model_rf.joblib")
print("\nModel saved as exam_model_rf.joblib")

# --- 4. 評估模型 (跟隊友保持一樣的指標) ---
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mse)

print("\n======================")
print("RF TEST RESULTS")
print("======================")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)

# --- 5. 預測範例 ---
int_preds = np.rint(preds).astype(int)
print("\nSample predictions (integerized):")
for i in range(10):
    print(f"Predicted: {int_preds[i]} | Actual: {y_test[i]}")

# --- 6. 準確度分析 (複製隊友的邏輯) ---
# 這是為了跟他的 NN 模型做 Apple-to-Apple 的比較
thresholds = [0.01, 0.02, 0.03, 0.05, 0.10]
y_test_safe = np.where(y_test == 0, 1e-9, y_test) # 避免除以 0

print("\nAccuracy at different tolerance percentages:")
for t in thresholds:
    diff_percent = np.abs(int_preds - y_test_safe) / y_test_safe
    correct = np.sum(diff_percent <= t)
    total = len(y_test)
    acc = correct / total * 100
    print(f"±{int(t*100)}% accuracy: {correct}/{total}  ({acc:.2f}%)")

# --- 7. 繪圖 ---
# Random Forest 沒有 "Epoch" 的概念，所以不能畫 Loss vs Epoch
# 建議畫 "預測值 vs 真實值" 的散佈圖，這對回歸問題很有用
plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.5, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

# 也可以畫 Feature Importance (這只有 RF 有，NN 沒有，是你的優勢！)
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 取前 10 個最重要的特徵
    top_k = 10
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances")
    plt.bar(range(top_k), importances[indices[:top_k]], align="center")
    # 如果你有 column names 可以放進去，這裡先用 index
    plt.xticks(range(top_k), [X_train.columns[i] for i in indices[:top_k]], rotation=45)
    plt.xlim([-1, top_k])
    plt.tight_layout()
    plt.show()