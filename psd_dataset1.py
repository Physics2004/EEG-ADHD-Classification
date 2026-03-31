import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

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

fs = 256  # sampling frequency

# =========================
# FIX SHAPE
# =========================
X = np.squeeze(X)  # (N, 2, 512)

print("New shape:", X.shape)

# =========================
# COMPUTE PSD
# =========================
psd_class0 = []
psd_class1 = []

for i in range(len(X)):
    signal = X[i]  # (2, 512)

    # 🔥 ensure correct dtype
    signal = np.array(signal, dtype=np.float64)

    # 🔥 average channels → (512,)
    signal = np.mean(signal, axis=0)

    # 🔥 force 1D
    signal = signal.flatten()

    # 🔥 safety check
    if len(signal) < 10:
        continue

    f, Pxx = welch(signal, fs=fs, nperseg=256)

    if y[i] == 1:
        psd_class1.append(Pxx)
    else:
        psd_class0.append(Pxx)

# =========================
# AVERAGE PSD
# =========================
psd_class0 = np.mean(psd_class0, axis=0)
psd_class1 = np.mean(psd_class1, axis=0)

# =========================
# PLOT
# =========================
plt.figure()
plt.semilogy(f, psd_class0, label="Class 0")
plt.semilogy(f, psd_class1, label="Class 1")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("PSD Comparison (ADHD Dataset)")
plt.legend()
plt.grid()

plt.savefig(os.path.join(save_path, "psd_plot_adhd.png"))
