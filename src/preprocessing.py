"""
===============================================================
🧱 Preprocessing Pipeline for Human Activity Recognition
===============================================================
Author: Hamady GACKOU
Master 2 - Machine Learning for Data Science (Université Paris Cité)

Purpose:
--------
This script implements a reproducible preprocessing pipeline that:
1. Loads raw sensor data (.txt files)
2. Builds a 3D tensor (347, 128, 9)
3. Extracts statistical features (mean, std, min, max, skewness, kurtosis)
4. Normalizes the resulting feature matrix
5. Saves `features.npy` and `labels.npy` for later clustering analysis
"""

# ============================================================
# Imports
# ============================================================
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

# ============================================================
# Utility Functions
# ============================================================

def load_sensor_files(data_path: str):
    """
    Load all 9 sensor variable files and return them as a list of matrices.
    Each file is expected to have shape (347, 128).
    """
    variable_names = [
        "accm_x", "accm_y", "accm_z",
        "acce_x", "acce_y", "acce_z",
        "vit_x", "vit_y", "vit_z"
    ]
    
    data = []
    for var in variable_names:
        file_path = os.path.join(data_path, f"{var}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        matrix = np.loadtxt(file_path)
        data.append(matrix)
    
    X_tensor = np.stack(data, axis=2)  # (347, 128, 9)
    print(f"Loaded 9 sensor files successfully → Tensor shape: {X_tensor.shape}")
    return X_tensor, variable_names


def load_labels(data_path: str):
    """
    Load activity labels (lab.txt) as integers.
    """
    label_path = os.path.join(data_path, "lab.txt")
    if not os.path.exists(label_path):
        raise FileNotFoundError("Missing file: lab.txt")
    y = np.loadtxt(label_path).astype(int)
    print(f" Labels loaded → Shape: {y.shape}, Classes: {np.unique(y)}")
    return y


def extract_statistical_features(X_tensor: np.ndarray):
    """
    Extract statistical features (mean, std, min, max, skewness, kurtosis)
    for each window (347) and each variable (9).
    Output: feature matrix of shape (347, 54)
    """
    n_windows, _, n_vars = X_tensor.shape
    features = np.zeros((n_windows, n_vars * 6))

    for i in range(n_windows):
        feats = []
        for j in range(n_vars):
            serie = X_tensor[i, :, j]
            feats.extend([
                np.mean(serie),
                np.std(serie),
                np.min(serie),
                np.max(serie),
                skew(serie),
                kurtosis(serie)
            ])
        features[i, :] = feats

    print(f"Extracted statistical features → Shape: {features.shape}")
    return features


def normalize_features(X_features: np.ndarray):
    """
    Normalize features (zero mean, unit variance) using StandardScaler.
    """
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_features)
    print(f" Normalization complete → Shape: {X_norm.shape}")
    return X_norm


# ============================================================
#  Main Pipeline Function
# ============================================================

def prepare_features(data_path: str, save_path: str = "../notebooks"):
    """
    Complete preprocessing pipeline:
      - Load data and labels
      - Build tensor
      - Extract statistical features
      - Normalize
      - Save .npy files
    """
    print(" Starting preprocessing pipeline...")
    
    # 1️ Load sensor data & labels
    X_tensor, variable_names = load_sensor_files(data_path)
    y = load_labels(data_path)
    
    # 2️ Extract statistical features
    X_features = extract_statistical_features(X_tensor)
    
    # 3️ Normalize features
    X_features_norm = normalize_features(X_features)
    
    # 4 Save to disk
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "features.npy"), X_features_norm)
    np.save(os.path.join(save_path, "labels.npy"), y)
    
    print(f" Saved preprocessed data in: {save_path}")
    print(" Preprocessing pipeline completed successfully.")
    return X_features_norm, y


# ============================================================
#  Run as a script (project-root execution)
# ============================================================
if __name__ == "__main__":
    # Always resolve paths relative to the project root
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # → src/
    PROJECT_ROOT = os.path.dirname(ROOT_DIR)               # → HumanActivityRecognition/
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")          # → HumanActivityRecognition/data
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notebooks")   # → HumanActivityRecognition/notebooks
    
    print(f" Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    X_features, y = prepare_features(DATA_DIR, OUTPUT_DIR)
