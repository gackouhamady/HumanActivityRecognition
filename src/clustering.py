"""
=====================================================================
STRICT MULTI-APPROACH CLUSTERING VALIDATION & EXECUTION
=====================================================================
Author: Hamady GACKOU
Université Paris Cité — Master 2 Machine Learning for Data Science
---------------------------------------------------------------------
Main driver script:
Checks, validates, and runs clustering for both:
1️⃣ Direct (Tensor-based) approach
2️⃣ Feature-based approach
=====================================================================
"""

# ============================================================
# Imports
# ============================================================
import os, sys, numpy as np

# Import both specialized modules
from src.feature_clustering import run_feature_clustering
from src.tensor_clustering import run_tensor_clustering

# ============================================================
# ✅ STEP 1 — PATH CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TENSOR_PATH = os.path.join(DATA_DIR, "X_direct.npy")
FEATURES_PATH = os.path.join(DATA_DIR, "features.npy")
LABELS_PATH = os.path.join(DATA_DIR, "labels.npy")

print(f"\n📂 Data directory: {DATA_DIR}")

# ============================================================
# ✅ STEP 2 — LOAD AND VALIDATE DATA
# ============================================================

def safe_load(path, name):
    """Load file safely, raise clear error if not found."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {name} file not found at: {path}")
    arr = np.load(path)
    print(f"✓ Loaded {name}: shape={arr.shape}")
    return arr

try:
    X_direct = safe_load(TENSOR_PATH, "Tensor (X_direct)")
    X_features = safe_load(FEATURES_PATH, "Feature Matrix (X_features)")
    y = safe_load(LABELS_PATH, "Labels (y)")
except Exception as e:
    print(f"\n🚫 Data loading failed: {e}")
    sys.exit(1)

# ============================================================
# ✅ STEP 3 — CONSISTENCY CHECKS
# ============================================================

def validate_shapes(X_direct, X_features, y):
    """Ensure both matrices have consistent structure and valid values."""
    problems = []
    if X_direct.ndim != 3:
        problems.append("X_direct must be 3D (n_samples, time_steps, variables)")
    if X_features.ndim != 2:
        problems.append("X_features must be 2D (n_samples, n_features)")
    if X_direct.shape[0] != X_features.shape[0] or X_features.shape[0] != y.shape[0]:
        problems.append("Number of samples must match across all datasets")
    if np.isnan(X_direct).any() or np.isnan(X_features).any():
        problems.append("NaN values detected in matrices")
    if np.isinf(X_direct).any() or np.isinf(X_features).any():
        problems.append("Infinite values detected in matrices")

    if problems:
        print("\n🚫 Validation failed:")
        for p in problems:
            print(f"   • {p}")
        sys.exit(1)
    else:
        print("\n✅ Data validation passed — all matrices are consistent.\n")

validate_shapes(X_direct, X_features, y)

# ============================================================
# ✅ STEP 4 — RUN CLUSTERING (Both Approaches)
# ============================================================
print("🚀 Starting Multi-Approach Clustering...\n")

try:
    feature_results = run_feature_clustering(X_features, n_clusters=6)
    tensor_results = run_tensor_clustering(X_direct, n_clusters=6)
except Exception as e:
    print(f"\n🚫 Clustering execution failed: {e}")
    sys.exit(1)

# ============================================================
# ✅ STEP 5 — SUMMARY OF RESULTS
# ============================================================
print("\n📊 Clustering Summary:")
for name, labels in {**feature_results, **tensor_results}.items():
    print(f"{name:20s} → clusters = {len(np.unique(labels))}")

print("\n🏁 Execution complete — both approaches ran successfully.")
