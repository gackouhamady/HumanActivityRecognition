# ============================================================
# Imports
# ============================================================
import numpy as np
import pandas as pd
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
        if c == -1:  # Ignore noise points (e.g. DBSCAN)
            continue
        idx = np.where(y_pred == c)[0]
        true_labels = y_true[idx]
        if len(true_labels) == 0:
            continue
        majority_label = np.bincount(true_labels).argmax()
        total_correct += np.sum(true_labels == majority_label)
    return total_correct / len(y_true)


def dunn_index(X, labels):
    """Compute Dunn Index (higher = better)."""
    clusters = np.unique(labels)
    clusters = [c for c in clusters if np.sum(labels == c) > 1]
    if len(clusters) < 2:
        return np.nan
    inter_dists, intra_dists = [], []
    for i, c1 in enumerate(clusters):
        cl1 = X[labels == c1]
        intra_dists.append(np.max(cdist(cl1, cl1)))
        for c2 in clusters[i + 1:]:
            cl2 = X[labels == c2]
            inter_dists.append(np.min(cdist(cl1, cl2)))
    return np.min(inter_dists) / np.max(intra_dists)


def sd_index(X, labels):
    """Simplified SD-Index (lower = better)."""
    clusters = np.unique(labels)
    clusters = [c for c in clusters if np.sum(labels == c) > 1]
    if len(clusters) < 2:
        return np.nan
    centroids = np.array([X[labels == c].mean(axis=0) for c in clusters])
    inter = np.std(cdist(centroids, centroids))
    intra = np.mean([np.mean(cdist(X[labels == c], [centroids[i]]))
                     for i, c in enumerate(clusters)])
    return intra / inter


def xie_beni_index(X, labels):
    """Xie–Beni index (lower = better)."""
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
# Data Preparation
# ============================================================

def prepare_data_for_metrics(X):
    """If tensor (3D), flatten for metric computation."""
    if X.ndim == 3:
        n, t, v = X.shape
        X = X.reshape(n, t * v)
    return X


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_internal(X, labels):
    """Compute internal metrics."""
    X = prepare_data_for_metrics(X)
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
    metrics["n_clusters"] = len(np.unique(labels))
    return metrics


def evaluate_external(y_true, y_pred):
    """Compute external metrics if ground truth is available."""
    metrics = {}
    try:
        metrics["ARI"] = adjusted_rand_score(y_true, y_pred)
        metrics["NMI"] = normalized_mutual_info_score(y_true, y_pred)
        h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
        metrics["Homogeneity"], metrics["Completeness"], metrics["VMeasure"] = h, c, v
        metrics["FMI"] = fowlkes_mallows_score(y_true, y_pred)
        metrics["Purity"] = purity_score(y_true, y_pred)
    except Exception:
        pass
    return metrics


def evaluate_clustering(X, y_true, y_pred):
    """Combined evaluation for both internal & external metrics."""
    return {
        "Internal": evaluate_internal(X, y_pred),
        "External": evaluate_external(y_true, y_pred) if y_true is not None else {}
    }


# ============================================================
# Batch Comparison for Feature + Tensor Results
# ============================================================

def compare_all_methods(X_features, X_tensor, y_true, results_feature, results_tensor):
    """
    Compare clustering methods from both feature and tensor approaches.
    Supports reduced spaces (PCA, UMAP) and distance-based metrics (DTW, etc.).
    """
    records = []

    # --- Feature-based results ---
    for reduction, method_results in results_feature.items():
        for method_name, labels in method_results.items():
            X_input = X_features.copy()
            internal = evaluate_internal(X_input, labels)
            external = evaluate_external(y_true, labels) if y_true is not None else {}
            record = {**internal, **external}
            record["Approach"] = "Feature-Based"
            record["Reduction"] = reduction.upper()
            record["Method"] = method_name
            records.append(record)

    # --- Tensor-based results ---
    for method_name, labels in results_tensor.items():
        X_input = X_tensor.copy()
        internal = evaluate_internal(X_input, labels)
        external = evaluate_external(y_true, labels) if y_true is not None else {}
        record = {**internal, **external}
        record["Approach"] = "Time-Series"
        record["Reduction"] = "-"
        record["Method"] = method_name
        records.append(record)

    df = pd.DataFrame(records)
    order_cols = ["Approach", "Reduction", "Method"] + [c for c in df.columns if c not in ["Approach", "Reduction", "Method"]]
    return df[order_cols]


# ============================================================
# Example Execution
# ============================================================
if __name__ == "__main__":
    import os
    from src.feature_clustering import run_all_feature_experiments
    from src.tensor_clustering import run_tensor_clustering_advanced

    NOTEBOOKS_PATH = r"C:\Users\MLSD\Desktop\HumanActivityRecognition\notebooks"
    FEATURES_PATH = os.path.join(NOTEBOOKS_PATH, "features.npy")
    TENSOR_PATH = os.path.join(NOTEBOOKS_PATH, "X_direct.npy")
    LABELS_PATH = os.path.join(NOTEBOOKS_PATH, "labels.npy")

    print("\n📂 Loading data...")
    X_features = np.load(FEATURES_PATH)
    X_tensor = np.load(TENSOR_PATH)
    try:
        y_true = np.load(LABELS_PATH)
    except Exception:
        y_true = None
        print("⚠️ No ground-truth labels found — external metrics skipped.")

    print(f"✅ Feature matrix: {X_features.shape} | Tensor: {X_tensor.shape}")

    # --- Run clustering methods ---
    results_feature = run_all_feature_experiments(X_features, n_clusters=6)
    results_tensor = run_tensor_clustering_advanced(X_tensor, n_clusters=6)

    # --- Compare all results ---
    df_eval = compare_all_methods(X_features, X_tensor, y_true, results_feature, results_tensor)

    # --- Display results ---
    print("\n🏁 Evaluation Summary (All by Silhouette):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_eval.sort_values("Silhouette", ascending=False).round(4))


    # --- Save results ---
    OUTPUT_PATH = os.path.join(NOTEBOOKS_PATH, "clustering_evaluation.csv")
    df_eval.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📊 Results saved to: {OUTPUT_PATH}")
