# EEG-based ADHD Classification

## Overview
This project presents an end-to-end pipeline for classifying ADHD vs Control subjects using EEG signals and machine learning techniques.

The workflow includes signal preprocessing, feature extraction, dataset construction, and model evaluation.

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
- Power Spectral Density (PSD)
- Canonical Correlation Analysis (CCA)

### 3. Modeling
- Machine learning / deep learning models
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
adhd_inspect.py
build_dataset_ADHD.py
psd_dataset1.py
cca_dataset1.py
subjectwise_adhd.py


---

## Tech Stack
- Python
- NumPy, Pandas
- Scikit-learn / TensorFlow

---

## Key Contribution
This project demonstrates an end-to-end pipeline for EEG signal processing and classification, with emphasis on reproducibility and structured data handling.

---

## Author
Kashish Thakur
