import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import os

# =========================
#  PATHS
# =========================
BASE_PATH = "../../Data/raw/dataset1/ADHD_EEG/EEG/"
SAVE_PATH = "../../Data/processed/"

FILES = {
    "FADHD.mat": 1,  # ADHD
    "MADHD.mat": 1,  # ADHD
    "FC.mat": 0,     # Control
    "MC.mat": 0      # Control
}

SAMPLING_RATE = 256
WINDOW_SIZE = 512   # 2 sec
STRIDE = 256        # 50% overlap

# =========================
#  FILTERS
# =========================

# Notch Filter (50 Hz)
def notch_filter(data, fs=256, freq=50):
    b, a = iirnotch(freq/(fs/2), Q=30)
    return filtfilt(b, a, data)

# Bandpass Filter (1–40 Hz)
def bandpass_filter(data, low=1, high=40, fs=256):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, data)

# =========================
#  DATASET BUILDER
# =========================
X = []
y = []
subject_ids = []

for file, label in FILES.items():
    print(f"\nProcessing {file}")
    
    data = sio.loadmat(BASE_PATH + file)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    dataset = data[key]  # shape (1, 11)
    
    for task_idx in range(11):
        task = dataset[0, task_idx]  # (subjects, time, channels)
        num_subjects = task.shape[0]
        
        for subj_idx in range(num_subjects):
            
            #  Remove corrupted subject (given in paper)
            if file == "FADHD.mat" and subj_idx == 6:
                continue
            
            eeg = task[subj_idx]  # (time, channels)
            
            #  Transpose → (channels, time)
            eeg = eeg.T
            TARGET_LENGTH = 3840  # minimum across tasks

            # Crop to fixed length
            if eeg.shape[1] > TARGET_LENGTH:
                eeg = eeg[:, :TARGET_LENGTH]
            
            #  Apply filters
            for ch in range(eeg.shape[0]):
                eeg[ch] = notch_filter(eeg[ch])      # remove 50 Hz noise
                eeg[ch] = bandpass_filter(eeg[ch])   # keep 1–40 Hz
            
            #  Channel-wise normalization
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / \
                  (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            
            # Windowing
            total_time = eeg.shape[1]
            
            for start in range(0, total_time - WINDOW_SIZE, STRIDE):
                segment = eeg[:, start:start + WINDOW_SIZE]  # (2, 512)
                
                X.append(segment)
                y.append(label)
                subject_ids.append(f"{file}_S{subj_idx}")

# =========================
# CONVERT TO NUMPY
# =========================
X = np.array(X, dtype=np.float32)
y = np.array(y)
subject_ids = np.array(subject_ids)

print("\nBefore reshape:", X.shape)

# =========================
#  EEGNet FORMAT
# =========================
X = np.transpose(X, (0, 1, 2))  # already (N, 2, 512)
X = np.expand_dims(X, axis=-1)  # (N, 2, 512, 1)

print("Final Shape:", X.shape)
print("Labels:", y.shape)
print("Subjects:", subject_ids.shape)

# =========================
#  SAVE FILES
# =========================
os.makedirs(SAVE_PATH, exist_ok=True)

np.save(SAVE_PATH + "adhd_dataset_X.npy", X)
np.save(SAVE_PATH + "adhd_dataset_y.npy", y)
np.save(SAVE_PATH + "adhd_subject_ids.npy", subject_ids)

print("\n Dataset saved successfully!")
