"""
===============================================================
Cluster Evaluation Module
===============================================================
Author: Hamady GACKOU
Master 2 – Machine Learning for Data Science (Université Paris Cité)

Purpose:
--------
Quantitatively assess clustering quality using both internal
(unsupervised) and external (supervised) evaluation metrics.

Metrics implemented:
--------------------
Internal (no ground truth required):
 - Silhouette Score
 - Davies–Bouldin Index
 - Calinski–Harabasz Index
 - Dunn Index
 - SD-Index
 - Xie–Beni Index (for fuzzy or centroid-based methods)

External (requires ground truth labels):
 - Adjusted Rand Index (ARI)
 - Normalized Mutual Information (NMI)
 - Homogeneity, Completeness, V-Measure
 - Purity Score
 - Fowlkes–Mallows Index (FMI)
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_completeness_v_measure, fowlkes_mallows_score
)
from scipy.spatial.distance import cdist

# ============================================================
# Helper Functions
# ============================================================

def purity_score(y_true, y_pred):
    """Compute clustering purity."""
    clusters = np.unique(y_pred)
    total_correct = 0
    for c in clusters:
        if c == -1:
            continue  # ignore noise (e.g., DBSCAN)
        idx = np.where(y_pred == c)[0]
        true_labels = y_true[idx]
        if len(true_labels) == 0:
            continue
        majority_label = np.bincount(true_labels).argmax()
        total_correct += np.sum(true_labels == majority_label)
    return total_correct / len(y_true)


def dunn_index(X, labels):
    """
    Compute Dunn Index (ratio of minimum inter-cluster distance
    to maximum intra-cluster distance). Higher is better.
    """
    clusters = np.unique(labels)
    clusters = [c for c in clusters if np.sum(labels == c) > 1]
    if len(clusters) < 2:
        return np.nan

    inter_dists = []
    intra_dists = []

    for i, c1 in enumerate(clusters):
        cluster_1 = X[labels == c1]
        intra_dists.append(np.max(cdist(cluster_1, cluster_1)))
        for c2 in clusters[i + 1:]:
            cluster_2 = X[labels == c2]
            inter_dists.append(np.min(cdist(cluster_1, cluster_2)))

    return np.min(inter_dists) / np.max(intra_dists)


def sd_index(X, labels):
    """
    Simplified SD-Index: ratio between average intra-cluster
    dispersion and standard deviation of inter-cluster distances.
    Lower values indicate better separation.
    """
    clusters = np.unique(labels)
    clusters = [c for c in clusters if np.sum(labels == c) > 1]
    if len(clusters) < 2:
        return np.nan

    centroids = np.array([X[labels == c].mean(axis=0) for c in clusters])
    inter = np.std(cdist(centroids, centroids))
    intra = np.mean([np.mean(cdist(X[labels == c], [centroids[i]])) for i, c in enumerate(clusters)])
    return intra / inter


def xie_beni_index(X, labels):
    """
    Compute Xie–Beni index for crisp clustering.
    Lower values indicate better compactness and separation.
    """
    clusters = np.unique(labels)
    clusters = [c for c in clusters if np.sum(labels == c) > 1]
    if len(clusters) < 2:
        return np.nan

    centroids = np.array([X[labels == c].mean(axis=0) for c in clusters])
    intra = np.sum([np.sum((X[labels == c] - centroids[i])**2)
                    for i, c in enumerate(clusters)]) / X.shape[0]
    inter = np.min(cdist(centroids, centroids)[np.nonzero(cdist(centroids, centroids))])
    return intra / inter

# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_internal(X, labels):
    """
    Compute internal clustering validation metrics.
    Returns a dictionary of results.
    """
    metrics = {}
    try:
        metrics["Silhouette"] = silhouette_score(X, labels)
    except Exception:
        metrics["Silhouette"] = np.nan
    try:
        metrics["DaviesBouldin"] = davies_bouldin_score(X, labels)
    except Exception:
        metrics["DaviesBouldin"] = np.nan
    try:
        metrics["CalinskiHarabasz"] = calinski_harabasz_score(X, labels)
    except Exception:
        metrics["CalinskiHarabasz"] = np.nan

    metrics["Dunn"] = dunn_index(X, labels)
    metrics["SDIndex"] = sd_index(X, labels)
    metrics["XieBeni"] = xie_beni_index(X, labels)

    return metrics


def evaluate_external(y_true, y_pred):
    """
    Compute external clustering validation metrics using ground truth labels.
    Returns a dictionary of results.
    """
    metrics = {}
    metrics["ARI"] = adjusted_rand_score(y_true, y_pred)
    metrics["NMI"] = normalized_mutual_info_score(y_true, y_pred)
    metrics["Homogeneity"], metrics["Completeness"], metrics["VMeasure"] = homogeneity_completeness_v_measure(y_true, y_pred)
    metrics["FMI"] = fowlkes_mallows_score(y_true, y_pred)
    metrics["Purity"] = purity_score(y_true, y_pred)
    return metrics


def evaluate_clustering(X, y_true, y_pred):
    """
    Combined evaluation: compute both internal and external metrics.
    Returns a nested dictionary.
    """
    results = {
        "Internal": evaluate_internal(X, y_pred),
        "External": evaluate_external(y_true, y_pred) if y_true is not None else None
    }
    return results


# ============================================================
# Batch Evaluation for Multiple Methods
# ============================================================

def compare_all_methods(X, y_true, results_dict):
    """
    Evaluate multiple clustering results stored in a dictionary:
      results_dict = { 'KMeans': labels1, 'GMM': labels2, ... }

    Returns a pandas DataFrame of all metrics.
    """
    import pandas as pd
    records = []

    for method, y_pred in results_dict.items():
        internal = evaluate_internal(X, y_pred)
        external = evaluate_external(y_true, y_pred) if y_true is not None else {}
        combined = {**internal, **external}
        combined["Method"] = method
        records.append(combined)

    df = pd.DataFrame(records)
    cols = ["Method"] + [c for c in df.columns if c != "Method"]
    df = df[cols]
    return df


# ============================================================
# Script Example
# ============================================================

if __name__ == "__main__":
    import os
    import pandas as pd

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FEATURES_PATH = os.path.join(PROJECT_ROOT, "notebooks", "features.npy")
    LABELS_PATH = os.path.join(PROJECT_ROOT, "notebooks", "labels.npy")

    X = np.load(FEATURES_PATH)
    y_true = np.load(LABELS_PATH)

    # Example: fake clustering results for demonstration
    from sklearn.cluster import KMeans
    y_pred = KMeans(n_clusters=6, random_state=42).fit_predict(X)

    # Evaluate
    results = evaluate_clustering(X, y_true, y_pred)
    print("Internal metrics:")
    for k, v in results["Internal"].items():
        print(f"  {k:20s}: {v:.4f}")
    print("\nExternal metrics:")
    for k, v in results["External"].items():
        print(f"  {k:20s}: {v:.4f}")

    # Example multi-method comparison
    results_dict = {"KMeans": y_pred, "GMM": y_pred}
    df_results = compare_all_methods(X, y_true, results_dict)
    print("\nSummary Table:\n", df_results)
