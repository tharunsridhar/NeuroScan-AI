import os
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive

from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.convnext import preprocess_input

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

drive.mount("/content/drive")

TRAIN_DIR = "/content/drive/MyDrive/Project work/Dataset/Epic and CSCR hospital Dataset_clean/Train"
TEST_DIR  = "/content/drive/MyDrive/Project work/Dataset/Epic and CSCR hospital Dataset_clean/Test"

MODEL_PATH = "/content/drive/MyDrive/Models/convnext_tiny_tumor.keras"

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

train_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_raw.class_names
NUM_CLASSES = len(class_names)

print("Classes:", class_names)

test_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.15)
])

AUTOTUNE = tf.data.AUTOTUNE

def preprocess(x, y):
    x = preprocess_input(x)
    return x, y

def augment_and_preprocess(x, y):
    x = augment(x, training=True)
    x = preprocess_input(x)
    return x, y

train_ds = train_raw.map(
    augment_and_preprocess,
    num_parallel_calls=AUTOTUNE
)

val_ds = val_raw.map(
    preprocess,
    num_parallel_calls=AUTOTUNE
)

test_ds = test_raw.map(
    preprocess,
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

labels = []

for _, y in train_raw:
    labels.extend(np.argmax(y.numpy(), axis=1))

labels = np.array(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(weights))

print("Class weights:", class_weights)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        "best_stage.h5",
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

base_model = ConvNeXtTiny(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))

x = base_model(inputs, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(256, activation="gelu")(x)
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

model.load_weights("best_stage.h5")
GRAPHS.append(hist_warmup)

print("Head Training...")

hist_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

model.load_weights("best_stage.h5")
GRAPHS.append(hist_head)

print("Fine-tune 50%...")

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

model.load_weights("best_stage.h5")
GRAPHS.append(hist_ft1)

best_val = max(hist_ft1.history["val_accuracy"])

print("Best Val:", best_val)

if best_val < TARGET_ACC:

    print("Full Fine-tuning...")

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
    GRAPHS.append(hist_ft2)
    model.load_weights("best_stage.h5")

def plot_history(histories):

    acc, val_acc, loss, val_loss = [], [], [], []

    for h in histories:

        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]

        loss += h.history["loss"]
        val_loss += h.history["val_loss"]

    e = range(1, len(acc)+1)

    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(e, acc, label="Train")
    plt.plot(e, val_acc, label="Val")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(e, loss, label="Train")
    plt.plot(e, val_loss, label="Val")
    plt.legend()
    plt.title("Loss")

    plt.show()

plot_history(GRAPHS)

model.evaluate(test_ds)

y_true = []
y_pred = []

for x, y in test_ds:

    p = model.predict(x, verbose=0)

    y_true.extend(np.argmax(y, axis=1))
    y_pred.extend(np.argmax(p, axis=1))

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

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

MODEL_PATH = "/content/drive/MyDrive/Project work/models/convnext_tiny_tumor.keras"

model.save(MODEL_PATH)

model.save(MODEL_PATH.replace(".keras", ".h5"))

print("Saved:", MODEL_PATH)
