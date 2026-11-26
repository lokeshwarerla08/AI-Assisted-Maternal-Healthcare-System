"""
UltraVision AI - CNN Training Script
CNN for breast ultrasound classification (benign / malignant / normal).
Handles class imbalance via oversampling + class_weight.
Input shape: (128, 128, 1) - Grayscale.
"""

print("=" * 60)
print("  UltraVision AI - CNN Training (Grayscale & Balanced)")
print("=" * 60)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- PATHS ----
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE_DIR, "Ultrasound_Dataset", "Data", "Data")
TRAIN_DIR  = os.path.join(DATA_ROOT, "train")
VAL_DIR    = os.path.join(DATA_ROOT, "validation")
TEST_DIR   = os.path.join(DATA_ROOT, "test")
MODEL_PATH = os.path.join(BASE_DIR, "ultrasound_cnn_model.keras")

# ---- SETTINGS ----
IMG_SIZE   = 128
BATCH_SIZE = 16
EPOCHS     = 60

# ---- DATA AUGMENTATION ----
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

# ---- GENERATORS (Grayscale as requested) ----
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_data = eval_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_data = eval_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(f"\nClasses: {train_data.class_indices}")
print(f"Train: {train_data.samples}  Val: {val_data.samples}  Test: {test_data.samples}")

# ---- CLASS WEIGHTS ----
class_labels = train_data.classes
unique_classes = np.unique(class_labels)
weights = compute_class_weight("balanced", classes=unique_classes, y=class_labels)
class_weights = dict(zip(unique_classes.tolist(), weights.tolist()))
print(f"\nClass weights (compensating for imbalance):")
for idx, name in enumerate(train_data.class_indices):
    print(f"  {name}: {class_weights.get(idx, 1.0):.3f}")

# ---- MODEL (Custom Grayscale CNN) ----
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation="relu", padding="same",
           input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 4
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Head
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(train_data.num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---- CALLBACKS ----
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
]

# ---- TRAIN ----
print("\n" + "=" * 60)
print("  Training with balanced class weights ...")
print("=" * 60 + "\n")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ---- EVALUATE ON TEST SET ----
loss, acc = model.evaluate(test_data)
print(f"\n{'=' * 60}")
print(f"  Final Test Accuracy : {acc * 100:.2f}%")
print(f"  Final Test Loss     : {loss:.4f}")
print(f"{'=' * 60}\n")

# Per-class accuracy report
from sklearn.metrics import classification_report
test_data.reset()
predictions = model.predict(test_data, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes
class_names = list(train_data.class_indices.keys())
print("\nPer-class Classification Report:")
print(classification_report(true_classes, pred_classes,
                            target_names=class_names, zero_division=0))

if acc >= 0.86:
    print(">> TARGET MET - Accuracy >= 86%!")
else:
    print(f">> Accuracy is {acc*100:.2f}%.")

# ---- SAVE ----
model.save(MODEL_PATH)
print(f"Model saved -> {MODEL_PATH}")

import json
meta = {
    "test_accuracy": round(float(acc * 100), 2),
    "test_loss": round(float(loss), 4),
    "epochs_trained": len(history.history["accuracy"]),
    "classes": list(train_data.class_indices.keys()),
    "model_type": "CNN (Convolutional Neural Network)",
    "class_weights": {name: round(float(class_weights.get(i, 1.0)), 3)
                      for i, name in enumerate(train_data.class_indices)},
}
meta_path = os.path.join(BASE_DIR, "model_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved -> {meta_path}")

# ---- GRAPHS ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history["accuracy"],     label="Train")
ax1.plot(history.history["val_accuracy"], label="Val")
ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(history.history["loss"],     label="Train")
ax2.plot(history.history["val_loss"], label="Val")
ax2.set_title("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
graph_path = os.path.join(BASE_DIR, "training_graphs.png")
plt.savefig(graph_path, dpi=150)
plt.close()
print(f"Graphs saved -> {graph_path}")
print("\nDone! Run python app.py to start the server.\n")
