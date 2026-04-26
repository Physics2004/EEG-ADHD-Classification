# EEG-based ADHD Classification using deep learning
87.66% classification accuracy| EEG signal processing pipeline| CNN (EEGNet) based classifier| Python · MNE · Scikit-learn

## Overview
An end-to-end pipeline for classifying ADHD vs Control subjects from raw EEG signals using signal processing and machine learning. This project covers the full workflow: preprocessing → feature extraction → model training → evaluation.
Clinical motivation: ADHD affects ~5–10% of children globally. EEG-based automated classification offers a non-invasive, low-cost diagnostic aid compared to subjective clinical assessments.

---

## Dataset
- Multi-subject EEG dataset
- 2 channels
- Labels:
  - 0 → Control
  - 1 → ADHD

---

## Methodology

### 1. Preprocessing
- Signal cleaning and normalization
- Subject-wise data handling

### 2. Feature Extraction
- Power Spectral Density (PSD)- Extracts frequency-band power (delta, theta, alpha, beta) as features
- Canonical Correlation Analysis (CCA)- Used to find linear combinations of EEG channels that maximally correlate with reference signals

### 3. Classification
CNN-based classifier trained on extracted EEG features
Subject-wise cross-validation to prevent data leakage
Evaluation: accuracy, confusion matrix, ROC-AUC, loss curves
t-SNE used for latent space visualization

### 4. Modeling
- Deep learning models (CNN EEGNet)
- Training and validation pipeline

---

## Results

- Accuracy: **87.66%**
- Evaluation Metrics:
  - Confusion Matrix
  - ROC Curve
  - Loss Curve

## Sample output
Confusion matrix (confusion_matrix_1774509120)
ROC curve (roc_curve_1774509120)
Loss and validation curve (loss_curve_1774509120)
tsne visualization (tsne_1774509120)



---

## Project Structure
├── adhd_inspect.py          # Data inspection and preprocessing
├── build_dataset_ADHD.py    # Feature matrix construction
├── psd_dataset1.py          # PSD feature extraction
├── cca_dataset1.py          # CCA feature extraction
├── subjectwise_adhd.py      # Model training and evaluation
├── README.md
└── [output plots]           # Confusion matrix, ROC, loss curves, t-SNE


---

## Tech Stack
- Python
- NumPy, Pandas
- Scikit-learn / TensorFlow

---

## Key Contribution
This project demonstrates an end-to-end pipeline for EEG signal processing and classification, with emphasis on reproducibility and structured data handling.
Full EEG preprocessing pipeline (cleaning, normalization, epoching)
Multi-method feature extraction (PSD + CCA) for robust representation
Subject-wise validation to avoid subject-leakage bias
87.66% accuracy with clear evaluation metrics and visualizations

---

## Author
Kashish Thakur
MSc Physics, University of Mumbai

