from google.colab import drive
drive.mount("/content/drive")

# Core
import os
import numpy as np

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Sklearn
from sklearn.utils.class_weight import compute_class_weight

TRAIN_DIR = "/content/drive/MyDrive/Project work/Dataset/Epic and CSCR hospital Dataset_clean/Train"
TEST_DIR  = "/content/drive/MyDrive/Project work/Dataset/Epic and CSCR hospital Dataset_clean/Test"

MODEL_SAVE_PATH = "/content/drive/MyDrive/Project work/models/Tumor_v2s_clean.keras"

os.path.exists(TRAIN_DIR)
os.path.exists(TEST_DIR)

IMG_SIZE = 384
BATCH_SIZE = 16
SEED = 42

WARMUP_EPOCHS = 3
HEAD_EPOCHS   = 5
FT1_EPOCHS    = 6
FT2_EPOCHS    = 4

TARGET_ACC = 0.90
GRAPHS=[]

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names     #  FIX ADDED
NUM_CLASSES = len(class_names)

print("Classes:", class_names)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

augment = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.15),
    tf.keras.layers.RandomBrightness(0.1)

])
train_ds = train_ds.map(
    lambda x, y: (augment(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

labels = []

for _, y in train_ds:
    labels.extend(np.argmax(y.numpy(), axis=1))

labels = np.array(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(weights))

print("Class Weights:", class_weights)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        "best_temp.h5",
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.3,
        min_lr=1e-6
    )
]

base_model = EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))

x = preprocess_input(inputs)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(256, activation="swish")(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

metrics = [
    "accuracy",
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall()
]

print("Warmup...")

model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-3),
    loss=loss_fn,
    metrics=metrics
)

hist_warmup = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=WARMUP_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

model.load_weights("best_temp.h5")
GRAPHS.append(hist_warmup)

print("Head Training...")

model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-3),
    loss=loss_fn,
    metrics=metrics
)

hist_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

model.load_weights("best_temp.h5")
GRAPHS.append(hist_head)

print("Fine-Tune 50%...")

n = len(base_model.layers)
unfreeze = int(0.5 * n)

for layer in base_model.layers[unfreeze:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.AdamW(5e-5),
    loss=loss_fn,
    metrics=metrics
)

hist_ft1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FT1_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

model.load_weights("best_temp.h5")
GRAPHS.append(hist_ft1)

# @title
best_val = max(hist_ft1.history["val_accuracy"])

print("Best Val Acc:", best_val)

if best_val < TARGET_ACC:

    print("Running Full Fine-Tuning...")

    base_model.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(2e-5),
        loss=loss_fn,
        metrics=metrics
    )

    hist_ft2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FT2_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )

    model.load_weights("best_temp.h5")
    GRAPHS.append(hist_ft2)

else:

    print("Skipping FT2 (Accuracy  90%)")

# ===============================
# TRAINING GRAPH
# ===============================

import matplotlib.pyplot as plt

def plot_history(histories):

    train_acc = []
    val_acc   = []
    train_loss = []
    val_loss   = []

    for h in histories:

        train_acc  += h.history["accuracy"]
        val_acc    += h.history["val_accuracy"]

        train_loss += h.history["loss"]
        val_loss   += h.history["val_loss"]

    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(14,5))

 # Accuracy Plot
    plt.subplot(1,2,1)

    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()

 # Loss Plot
    plt.subplot(1,2,2)

    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()

    plt.show()

plot_history(GRAPHS)

# ===============================
# CONFUSION MATRIX
# ===============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

y_true = []
y_pred = []

# Get predictions
for x, y in test_ds:

    preds = model.predict(x, verbose=0)

    y_true.extend(np.argmax(y.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()

# Classification Report
print("Classification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

print("Evaluating on Test Set...")

model.evaluate(test_ds)

# Save in Keras format (recommended)
model.save(MODEL_SAVE_PATH)

# Save in H5 format (backup)
H5_PATH = MODEL_SAVE_PATH.replace(".keras", ".h5")
model.save(H5_PATH)

print("Model saved at:")
print("Keras :", MODEL_SAVE_PATH)
print("H5    :", H5_PATH)
