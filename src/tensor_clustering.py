
# ============================================================
# Imports
# ============================================================
import numpy as np
import warnings

# Try to import tslearn modules (optional dependency)
try:
    from tslearn.clustering import TimeSeriesKMeans, KShape
    from tslearn.barycenters import dtw_barycenter_averaging
    from tslearn.metrics import cdist_dtw, cdist_soft_dtw, cdist_soft_dtw_normalized
except ImportError:
    TimeSeriesKMeans = None
    KShape = None
    warnings.warn("⚠️ tslearn not installed. Run: pip install tslearn")

# ============================================================
# Helper — Summary print
# ============================================================
def print_clusters(name, labels):
    """Utility print for summary of cluster results."""
    print(f"✅ {name:25s} → clusters = {len(np.unique(labels))}")


# ============================================================
# Time-Series Clustering Configurations
# ============================================================
def run_euclidean_kmeans(X, n_clusters=6, random_state=42):
    """Classic KMeans using Euclidean distance."""
    if TimeSeriesKMeans is None:
        raise ImportError("TimeSeriesKMeans not available.")
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=random_state)
    labels = model.fit_predict(X)
    print_clusters("Euclidean_KMeans", labels)
    return labels


def run_dtw_kmeans(X, n_clusters=6, random_state=42):
    """DTW-based KMeans — robust to temporal shifts."""
    if TimeSeriesKMeans is None:
        raise ImportError("TimeSeriesKMeans not available.")
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=random_state)
    labels = model.fit_predict(X)
    print_clusters("DTW_KMeans", labels)
    return labels


def run_softdtw_kmeans(X, n_clusters=6, gamma=0.5, random_state=42):
    """Soft-DTW KMeans (differentiable, smoother DTW)."""
    if TimeSeriesKMeans is None:
        raise ImportError("TimeSeriesKMeans not available.")
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="softdtw",
        metric_params={"gamma": gamma},
        random_state=random_state
    )
    labels = model.fit_predict(X)
    print_clusters("SoftDTW_KMeans", labels)
    return labels


def run_kshape(X, n_clusters=6, random_state=42):
    """KShape — correlation-based clustering (phase invariant)."""
    if KShape is None:
        raise ImportError("KShape not available.")
    model = KShape(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    print_clusters("KShape", labels)
    return labels


def run_dba_kmeans(X, n_clusters=6, random_state=42, max_iter=50):
    """
    KMeans with DTW Barycenter Averaging (DBA) centers.
    Slow but provides realistic centroids under DTW alignment.
    """
    if TimeSeriesKMeans is None:
        raise ImportError("TimeSeriesKMeans not available.")
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        verbose=False,
        max_iter=max_iter,
        random_state=random_state,
        n_init=1,
    )
    labels = model.fit_predict(X)
    print_clusters("DBA_KMeans", labels)
    return labels


# ============================================================
# Wrapper — Execute all configurations
# ============================================================
def run_tensor_clustering_advanced(X_tensor, n_clusters=6):
    """
    Execute all time-series clustering configurations
    with different distance metrics.
    """
    print("\n🧩 Running Advanced Time-Series Clustering...\n")

    if TimeSeriesKMeans is None or KShape is None:
        raise ImportError("⚠️ tslearn must be installed. Run: pip install tslearn")

    results = {}
    methods = {
        "Euclidean_KMeans": run_euclidean_kmeans,
        "DTW_KMeans": run_dtw_kmeans,
        "SoftDTW_KMeans": run_softdtw_kmeans,
        "KShape": run_kshape,
        "DBA_KMeans": run_dba_kmeans,
    }

    for name, func in methods.items():
        try:
            labels = func(X_tensor, n_clusters)
            results[name] = labels
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    print(f"\n✅ {len(results)} tensor clustering configurations executed successfully.\n")
    return results


# ============================================================
# Module exports (for importability)
# ============================================================
__all__ = [
    "run_tensor_clustering_advanced",
    "run_euclidean_kmeans",
    "run_dtw_kmeans",
    "run_softdtw_kmeans",
    "run_kshape",
    "run_dba_kmeans",
]
