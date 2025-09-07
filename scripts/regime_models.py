try:
    import hdbscan
except ImportError:
    import os
    os.system('pip install hdbscan')
    import hdbscan



import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.decomposition import PCA


def apply_kmeans(features: pd.DataFrame, n_clusters: int = 3):
    """
    Applies KMeans clustering for regime detection.

    Returns:
    - DataFrame with regime labels
    - KMeans model
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    regimes = model.fit_predict(X_scaled)
    return pd.Series(regimes, index=features.index, name='regime_kmeans'), model


def apply_gmm(features: pd.DataFrame, n_components: int = 3):
    """
    Gaussian Mixture Model clustering for probabilistic regime detection.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    return pd.Series(labels, index=features.index, name='regime_gmm'), gmm


def apply_hdbscan(features: pd.DataFrame, min_cluster_size: int = 30):
    """
    Applies HDBSCAN for density-based regime detection.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X_scaled)
    return pd.Series(labels, index=features.index, name='regime_hdbscan'), model


def reduce_dimensionality(features: pd.DataFrame, method='pca', n_components=2):
    """
    Optional dimensionality reduction for visualization.
    """
    if method == 'pca':
        model = PCA(n_components=n_components)
        reduced = model.fit_transform(features)
        return pd.DataFrame(reduced, index=features.index, columns=[f'PC{i+1}' for i in range(n_components)])
    return None
