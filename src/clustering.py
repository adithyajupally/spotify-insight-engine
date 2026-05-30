"""
src/clustering.py
-----------------
K-Means clustering and PCA dimensionality reduction for songs.
All functions are stateless: arrays and DataFrames are passed in and
fitted objects are returned explicitly so the notebook and Streamlit
app can share the same artefacts.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Default cluster-name mapping
# ---------------------------------------------------------------------------

DEFAULT_CLUSTER_NAMES: Dict[int, str] = {
    0: "Party Songs",
    1: "Workout Songs",
    2: "Calm Acoustic",
    3: "Emotional Songs",
    4: "Relaxing Songs",
}


# ---------------------------------------------------------------------------
# Elbow method
# ---------------------------------------------------------------------------


def plot_elbow(
    X_cluster: np.ndarray,
    k_range: range | None = None,
    random_state: int = 42,
) -> None:
    """Fit K-Means for a range of K values and plot the elbow curve.

    Parameters
    ----------
    X_cluster:
        Scaled feature matrix (e.g. from
        :func:`preprocessing.build_cluster_matrix`).
    k_range:
        Range of K values to try. Defaults to ``range(1, 11)``.
    random_state:
        Random seed for K-Means.
    """
    if k_range is None:
        k_range = range(1, 11)

    inertia: List[float] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X_cluster)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertia, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# K-Means fitting
# ---------------------------------------------------------------------------


def fit_kmeans(
    df_cluster: pd.DataFrame,
    X_cluster: np.ndarray,
    n_clusters: int = 5,
    cluster_names: Dict[int, str] | None = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans]:
    """Fit K-Means and append ``cluster`` / ``cluster_name`` columns.

    Parameters
    ----------
    df_cluster:
        Sampled DataFrame (output of
        :func:`preprocessing.build_cluster_matrix`). A copy is made
        internally so the original is not mutated.
    X_cluster:
        Scaled feature matrix corresponding to *df_cluster*.
    n_clusters:
        Number of clusters K.
    cluster_names:
        Dict mapping integer cluster labels to human-readable names.
        Defaults to :data:`DEFAULT_CLUSTER_NAMES`.
    random_state:
        Random seed.

    Returns
    -------
    Tuple[pd.DataFrame, KMeans]
        - *df_cluster*: DataFrame with ``'cluster'`` and
          ``'cluster_name'`` columns added.
        - *kmeans*: The fitted :class:`~sklearn.cluster.KMeans` model.
    """
    if cluster_names is None:
        cluster_names = DEFAULT_CLUSTER_NAMES

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    df_cluster = df_cluster.copy()
    df_cluster["cluster"] = kmeans.fit_predict(X_cluster)
    df_cluster["cluster_name"] = df_cluster["cluster"].map(cluster_names)

    print(f"K-Means fitted with K={n_clusters}.")
    print(df_cluster[["cluster", "cluster_name"]].value_counts().to_string())
    return df_cluster, kmeans


# ---------------------------------------------------------------------------
# PCA visualisation
# ---------------------------------------------------------------------------


def plot_pca_clusters(
    df_cluster: pd.DataFrame,
    X_cluster: np.ndarray,
    n_components: int = 2,
) -> Tuple[PCA, np.ndarray]:
    """Reduce *X_cluster* to 2-D with PCA and show a scatter plot.

    Parameters
    ----------
    df_cluster:
        DataFrame with a ``'cluster_name'`` column
        (output of :func:`fit_kmeans`).
    X_cluster:
        Scaled feature matrix.
    n_components:
        Number of PCA components (must be 2 for a 2-D scatter).

    Returns
    -------
    Tuple[PCA, np.ndarray]
        - *pca*: The fitted :class:`~sklearn.decomposition.PCA` object.
        - *X_pca*: The transformed (n_samples × n_components) array.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_cluster)

    fig = px.scatter(
        df_cluster,
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color="cluster_name",
        title="PCA Visualisation of Song Clusters",
        labels={
            "x": "First Principal Component",
            "y": "Second Principal Component",
        },
        hover_data=["track_name", "artists"],
    )
    fig.show()

    explained = pca.explained_variance_ratio_
    print(f"Explained variance by component: {explained}")
    return pca, X_pca


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_kmeans(
    kmeans: KMeans,
    path: str = "models/kmeans.pkl",
) -> None:
    """Serialize *kmeans* with joblib.

    Parameters
    ----------
    kmeans:
        A fitted :class:`~sklearn.cluster.KMeans` model.
    path:
        Destination ``.pkl`` path (parent dirs are created if needed).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(kmeans, path)
    print(f"K-Means model saved to '{path}'")