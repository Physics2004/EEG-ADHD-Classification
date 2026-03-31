import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# =========================
#  LOAD DATA
# =========================
BASE_PATH = "../../Data/processed/"

X = np.load(BASE_PATH + "adhd_dataset_X.npy")
y = np.load(BASE_PATH + "adhd_dataset_y.npy")
subject_ids = np.load(BASE_PATH + "adhd_subject_ids.npy")

print("Data shape:", X.shape)
print("Subjects:", len(np.unique(subject_ids)))

# =========================
#  SUBJECT-WISE SPLIT
# =========================
unique_subjects = np.unique(subject_ids)

train_subj, test_subj = train_test_split(
    unique_subjects, test_size=0.2, random_state=42
)

train_mask = np.isin(subject_ids, train_subj)
test_mask = np.isin(subject_ids, test_subj)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("\nTrain subjects:", len(train_subj))
print("Test subjects:", len(test_subj))
print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# =========================
# ⚖️CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# =========================
#  CREATE SAVE FOLDER
# =========================
SAVE_DIR = "../../Scripts/training/plots_adhd/"
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = str(int(time.time()))

# =========================
#  EEGNet MODEL
# =========================
def EEGNet():
    input_shape = (2, 512, 1)

    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(16, (1, 64), padding='same')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.DepthwiseConv2D((2, 1), depth_multiplier=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)

    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(0.5)(x)

    # Block 2
    x = layers.Conv2D(32, (1, 16), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)

    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(0.5)(x)

    # Classification
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='elu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# =========================
# TRAIN MODEL
# =========================
model = EEGNet()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# =========================
#  EVALUATION
# =========================
probs = model.predict(X_test)
preds = np.argmax(probs, axis=1)

print("\n========== SUBJECT-WISE RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))

# =========================
#  LOSS CURVE (TRAIN + VAL)
# =========================
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(SAVE_DIR + f"loss_curve_{timestamp}.png", dpi=300)
plt.close()

# =========================
#  CONFUSION MATRIX PLOT
# =========================
cm = confusion_matrix(y_test, preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(SAVE_DIR + f"confusion_matrix_{timestamp}.png", dpi=300)
plt.close()

# =========================
#  ROC CURVE
# =========================
probs = probs[:, 1]

fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(SAVE_DIR + f"roc_curve_{timestamp}.png", dpi=300)
plt.close()

# =========================
#  t-SNE VISUALIZATION
# =========================
X_flat = X_test.reshape(X_test.shape[0], -1)

if X_flat.shape[0] > 2000:
    X_flat = X_flat[:2000]
    y_vis = y_test[:2000]
else:
    y_vis = y_test

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_flat)

plt.figure()
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_vis, cmap='coolwarm', s=10)
plt.colorbar(scatter)
plt.title("t-SNE Visualization")
plt.savefig(SAVE_DIR + f"tsne_{timestamp}.png", dpi=300)
plt.close()

print(f"\n All plots saved in: {SAVE_DIR}")
