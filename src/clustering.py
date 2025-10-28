"""
===============================================================
Clustering Methods Implementation
===============================================================
Author: Hamady GACKOU
Master 2 – Machine Learning for Data Science (Université Paris Cité)

Purpose:
--------
Apply and compare multiple clustering algorithms for Human Activity Recognition.
Supports both:
 - Statistical feature matrix (347×54)
 - Temporal tensor data (347×128×9)

Methods implemented (>10 total):
--------------------------------
1. K-Means
2. Agglomerative Hierarchical Clustering (CAH)
3. DBSCAN
4. HDBSCAN
5. Gaussian Mixture Models (GMM)
6. Spectral Clustering
7. Birch
8. Affinity Propagation
9. OPTICS
10. MiniBatch K-Means
11. Self-Organizing Map (SOM)
12. DTW-based K-Medoids (for time-series tensor)
"""

# ============================================================
#  Imports
# ============================================================
import numpy as np
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering,
    Birch, AffinityPropagation, OPTICS, MiniBatchKMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import warnings

# Optional advanced packages
try:
    import hdbscan
    from tslearn.clustering import TimeSeriesKMeans
    from minisom import MiniSom
except ImportError:
    warnings.warn("Some optional packages (hdbscan, tslearn, minisom) not installed.")


# ============================================================
#  Helper Functions
# ============================================================

def normalize_features(X):
    """Standardize features for clustering."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def reshape_tensor_to_features(X_tensor):
    """Flatten a 3D tensor (n_samples, time_steps, variables) into 2D."""
    n, t, v = X_tensor.shape
    return X_tensor.reshape(n, t * v)


# ============================================================
# Clustering Methods (Feature Matrix)
# ============================================================

def run_kmeans(X, n_clusters=6, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    return model.fit_predict(X)

def run_minibatch_kmeans(X, n_clusters=6, random_state=42):
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=32)
    return model.fit_predict(X)

def run_hierarchical(X, n_clusters=6, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(X)

def run_dbscan(X, eps=0.7, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

def run_hdbscan(X, min_cluster_size=5):
    """Requires hdbscan library."""
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return model.fit_predict(X)

def run_gmm(X, n_clusters=6, random_state=42):
    model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    return model.fit_predict(X)

def run_spectral(X, n_clusters=6, random_state=42):
    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=random_state)
    return model.fit_predict(X)

def run_birch(X, n_clusters=6):
    model = Birch(n_clusters=n_clusters)
    return model.fit_predict(X)

def run_affinity(X, damping=0.9):
    model = AffinityPropagation(damping=damping)
    return model.fit_predict(X)

def run_optics(X, min_samples=10, xi=0.05):
    model = OPTICS(min_samples=min_samples, xi=xi)
    return model.fit_predict(X)


# ============================================================
# Self-Organizing Map (SOM)
# ============================================================

def run_som(X, som_x=10, som_y=10, sigma=1.0, learning_rate=0.5, num_iteration=1000, random_state=42):
    """
    Train a Self-Organizing Map on normalized feature space.
    Requires `minisom` package.
    """
    from minisom import MiniSom
    np.random.seed(random_state)
    
    som = MiniSom(som_x, som_y, X.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(X)
    som.train_random(X, num_iteration)
    
    # Map each sample to its winning neuron
    winners = np.array([som.winner(x) for x in X])
    labels = np.array([wx * som_y + wy for wx, wy in winners])
    print(f"✅ SOM trained → grid {som_x}×{som_y}, total neurons={som_x*som_y}")
    return labels


# ============================================================
# DTW-based Clustering (Time-Series Tensor)
# ============================================================

def run_dtw_kmeans(X_tensor, n_clusters=6, metric="dtw", random_state=42):
    """
    Dynamic Time Warping-based K-Means using tslearn.
    Works on 3D tensors: (n_samples, time_steps, variables)
    """
    from tslearn.clustering import TimeSeriesKMeans
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=random_state)
    return model.fit_predict(X_tensor)


def run_kshape(X_tensor, n_clusters=6, random_state=42):
    """Shape-based clustering for time-series (tslearn)."""
    from tslearn.clustering import KShape
    model = KShape(n_clusters=n_clusters, random_state=random_state)
    return model.fit_predict(X_tensor)


# ============================================================
# Wrapper for Multi-Method Comparison
# ============================================================

def run_all_clustering_methods(X_features, X_tensor=None, n_clusters=6):
    """
    Run multiple clustering algorithms and return results in a dictionary.
    X_features: 2D statistical features (347×54)
    X_tensor: 3D tensor (347×128×9) for time-series methods (optional)
    """
    print(" Running multiple clustering algorithms...")

    # Normalize features
    X = normalize_features(X_features)
    results = {}

    # Feature-space clustering
    results["KMeans"] = run_kmeans(X, n_clusters)
    results["MiniBatchKMeans"] = run_minibatch_kmeans(X, n_clusters)
    results["Hierarchical"] = run_hierarchical(X, n_clusters)
    results["GMM"] = run_gmm(X, n_clusters)
    results["Spectral"] = run_spectral(X, n_clusters)
    results["Birch"] = run_birch(X, n_clusters)
    results["Affinity"] = run_affinity(X)
    results["DBSCAN"] = run_dbscan(X)
    try:
        results["HDBSCAN"] = run_hdbscan(X)
    except Exception as e:
        print("⚠️ HDBSCAN failed:", e)
    try:
        results["SOM"] = run_som(X)
    except Exception as e:
        print("⚠️ SOM failed:", e)
    try:
        if X_tensor is not None:
            results["DTW_KMeans"] = run_dtw_kmeans(X_tensor, n_clusters)
            results["KShape"] = run_kshape(X_tensor, n_clusters)
    except Exception as e:
        print("⚠️ Time-series clustering failed:", e)

    print(f"✅ {len(results)} clustering results obtained.")
    return results


# ============================================================
#  Run as standalone script
# ============================================================
if __name__ == "__main__":
    import os

    # Load saved features and labels
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FEATURES_PATH = os.path.join(PROJECT_ROOT, "notebooks", "features.npy")
    LABELS_PATH = os.path.join(PROJECT_ROOT, "notebooks", "labels.npy")
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")

    X_features = np.load(FEATURES_PATH)
    y_true = np.load(LABELS_PATH)

    # Optional: load tensor if needed for DTW / KShape
    try:
        from preprocessing import load_sensor_files
        X_tensor, _ = load_sensor_files(DATA_PATH)
    except Exception:
        X_tensor = None

    # Run all methods
    all_results = run_all_clustering_methods(X_features, X_tensor, n_clusters=6)

    # Print cluster label shapes
    for name, labels in all_results.items():
        print(f"{name:20s} → clusters found: {len(np.unique(labels))}")
