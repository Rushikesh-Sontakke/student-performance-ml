import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.flatten()

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.flatten()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

model = Sequential([

    Dense(256, activation="relu", kernel_initializer='he_normal',
          kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    BatchNormalization(),

    Dense(128, activation="relu", kernel_initializer='he_normal',
          kernel_regularizer=l2(0.001)),
    BatchNormalization(),

    Dense(64, activation="relu", kernel_initializer='he_normal',
          kernel_regularizer=l2(0.001)),
    BatchNormalization(),

    Dense(32, activation="relu", kernel_initializer='he_normal',
          kernel_regularizer=l2(0.001)),
    BatchNormalization(),

    Dense(1)  
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

lr_schedule = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=16,
    callbacks=[early_stop, lr_schedule],
    verbose=1
)

model.save("exam_model.h5")
print("\nModel saved as exam_model.h5")

model.save("exam_model.keras")
print("Model saved as exam_model.keras")

model.export("exam_model_saved")
print("SavedModel exported to exam_model_saved/")

mse, mae = model.evaluate(X_test, y_test, verbose=0)
rmse = np.sqrt(mse)

print("\n======================")
print("TEST RESULTS")
print("======================")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)

preds = model.predict(X_test).flatten()

int_preds = np.rint(preds).astype(int)

print("\nSample predictions (integerized):")
for i in range(10):
    print(f"Predicted: {int_preds[i]} | Actual: {y_test[i]}")

thresholds = [0.01, 0.02, 0.03, 0.05, 0.10]
y_test_safe = np.where(y_test == 0, 1e-9, y_test)

print("\nAccuracy at different tolerance percentages:")
for t in thresholds:
    diff_percent = np.abs(int_preds - y_test_safe) / y_test_safe
    correct = np.sum(diff_percent <= t)
    total = len(y_test)
    acc = correct / total * 100
    print(f"±{int(t*100)}% accuracy: {correct}/{total}  ({acc:.2f}%)")

history_dict = history.history

loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
mae = history_dict["mae"]
val_mae = history_dict["val_mae"]
epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.title("MSE Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, mae, label="Training MAE")
plt.plot(epochs_range, val_mae, label="Validation MAE")
plt.title("MAE per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
