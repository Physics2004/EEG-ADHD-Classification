import os
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================
# CREATE SAVE FOLDER
# =========================
save_path = "plots"
os.makedirs(save_path, exist_ok=True)

# =========================
# LOAD DATA
# =========================
X = np.load("../../Data/processed/adhd_dataset_X.npy")
y = np.load("../../Data/processed/adhd_dataset_y.npy")

print("Loaded:", X.shape)

# =========================
# FIX SHAPE
# =========================
X = np.squeeze(X)   # (N, 2, 512)

print("New shape:", X.shape)

# =========================
# NORMALIZE (VERY IMPORTANT)
# =========================
X = (X - np.mean(X, axis=(1,2), keepdims=True)) / (np.std(X, axis=(1,2), keepdims=True) + 1e-8)

# =========================
# FLATTEN
# =========================
X = X.reshape(len(X), -1)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# =========================
# CCA
# =========================
cca = CCA(n_components=1)

y_train_2d = y_train.reshape(-1, 1)

X_train_c, _ = cca.fit_transform(X_train, y_train_2d)
X_test_c = cca.transform(X_test)

# =========================
# SVM CLASSIFIER
# =========================
clf = SVC(kernel='rbf')
clf.fit(X_train_c, y_train)

preds = clf.predict(X_test_c)

# =========================
# RESULTS
# =========================
accuracy = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

print("\nCCA ADHD Accuracy:", accuracy * 100)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, preds))

# =========================
# SAVE RESULTS
# =========================
with open(os.path.join(save_path, "cca_adhd_results.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

# =========================
# VISUALIZATION (1D)
# =========================
plt.figure()
plt.scatter(X_test_c[:, 0], np.zeros_like(X_test_c[:, 0]), c=y_test)
plt.title("CCA Projection (ADHD Dataset)")
plt.xlabel("Component 1")

plt.savefig(os.path.join(save_path, "cca_adhd_plot.png"))
