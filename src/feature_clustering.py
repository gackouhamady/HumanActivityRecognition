

# ============================================================
# Imports
# ============================================================
import numpy as np
import warnings
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, Birch,
    AffinityPropagation, OPTICS, MiniBatchKMeans, SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional dependencies
try:
    import hdbscan
except ImportError:
    hdbscan = None
    warnings.warn("⚠️ hdbscan not installed.")

try:
    from umap import UMAP
except ImportError:
    UMAP = None
    warnings.warn("⚠️ umap not installed.")

try:
    from minisom import MiniSom
except ImportError:
    MiniSom = None
    warnings.warn("⚠️ minisom not installed.")


# ============================================================
# Helpers
# ============================================================
def normalize(X):
    """Z-score normalization."""
    return StandardScaler().fit_transform(X)


def reduce_dimension(X, method="pca", n_components=10, random_state=42):
    """Apply PCA, UMAP, or t-SNE reduction."""
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "umap":
        if UMAP is None:
            raise ImportError("UMAP not available. Install with 'pip install umap-learn'.")
        reducer = UMAP(n_components=n_components, random_state=random_state)
    elif method in ["tsne", "t-sne", "tcna"]:
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    X_reduced = reducer.fit_transform(X)
    print(f"✅ {method.upper()} applied — new shape = {X_reduced.shape}")
    return X_reduced


# ============================================================
# Clustering Methods (Feature-Based)
# ============================================================
def run_kmeans(X, n_clusters=6):
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)

def run_minibatch_kmeans(X, n_clusters=6):
    return MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=32).fit_predict(X)

def run_hierarchical(X, n_clusters=6):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X)

def run_gmm(X, n_clusters=6):
    return GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(X)

def run_spectral(X, n_clusters=6):
    return SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42).fit_predict(X)

def run_birch(X, n_clusters=6):
    return Birch(n_clusters=n_clusters).fit_predict(X)

def run_affinity(X):
    return AffinityPropagation(damping=0.9).fit_predict(X)

def run_dbscan(X):
    return DBSCAN(eps=0.7, min_samples=5).fit_predict(X)

def run_hdbscan(X):
    if hdbscan is None:
        raise ImportError("hdbscan not available.")
    return hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X)

def run_optics(X):
    return OPTICS(min_samples=10, xi=0.05).fit_predict(X)

def run_som(X):
    if MiniSom is None:
        raise ImportError("MiniSom not available.")
    np.random.seed(42)
    som = MiniSom(10, 10, X.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, 1000)
    winners = np.array([som.winner(x) for x in X])
    return np.array([wx * 10 + wy for wx, wy in winners])


# ============================================================
# Wrapper — Run all clustering methods
# ============================================================
def run_all_feature_methods(X, n_clusters=6):
    """Run all clustering methods once."""
    results = {}
    methods = {
        "KMeans": run_kmeans,
        "MiniBatchKMeans": run_minibatch_kmeans,
        "CAH_Ward": run_hierarchical,
        "GMM": run_gmm,
        "Spectral": run_spectral,
        "Birch": run_birch,
        "Affinity": run_affinity,
        "DBSCAN": run_dbscan,
        "OPTICS": run_optics,
    }

    # Optional methods
    if hdbscan is not None:
        methods["HDBSCAN"] = run_hdbscan
    if MiniSom is not None:
        methods["SOM"] = run_som

    for name, func in methods.items():
        try:
            labels = func(X)
            results[name] = labels
            print(f"✅ {name:15s} → clusters = {len(np.unique(labels))}")
        except Exception as e:
            print(f"⚠️ {name} failed:", e)
    return results


# ============================================================
# MAIN EXECUTION — 4 scenarios (Baseline + PCA + UMAP + t-SNE)
# ============================================================
def run_all_feature_experiments(X_features, n_clusters=6):
    """
    Run clustering for:
    - baseline (no reduction)
    - PCA
    - UMAP
    - t-SNE
    """
    all_results = {}
    X_norm = normalize(X_features)

    for method in ["none", "pca", "umap", "tcna"]:
        print(f"\n{'='*60}\n🔹 Dimensionality Reduction: {method.upper()}\n{'='*60}")
        if method == "none":
            X_input = X_norm
        else:
            X_input = reduce_dimension(X_norm, method=method, n_components=10)

        results = run_all_feature_methods(X_input, n_clusters=n_clusters)
        all_results[method] = results

    print("\n🏁 All experiments finished successfully.")
    return all_results
