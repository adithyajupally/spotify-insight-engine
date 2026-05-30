"""
src/preprocessing.py
--------------------
Data loading, cleaning, feature engineering, scaling, and persistence.
All functions are pure (take DataFrames in, return DataFrames / artefacts out)
so they can be called from the notebook, the Streamlit app, or tests without
any shared global state.
"""

from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ---------------------------------------------------------------------------
# Column-name constants (single source of truth for the whole project)
# ---------------------------------------------------------------------------

AUDIO_FEATURES: list[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_mins",
]

REC_FEATURES: list[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "valence",
    "tempo",
]

CLUSTER_FEATURES: list[str] = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "tempo",
    "loudness",
    "speechiness",
    "instrumentalness",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_data(path: str = "data/dataset.csv") -> pd.DataFrame:
    """Load the raw Spotify CSV from *path*.

    Parameters
    ----------
    path:
        Relative or absolute path to ``dataset.csv``.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame exactly as read from disk.
    """
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null values and exact duplicate rows.

    Parameters
    ----------
    df:
        Raw DataFrame returned by :func:`load_raw_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (new object; original is unchanged).
    """
    before = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df):,} rows (nulls + duplicates). Remaining: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Add derived columns used by all downstream modules.

    New columns added (in-place on a copy):
    - ``popularity``     – normalised to [0, 1]
    - ``duration_mins``  – track duration in minutes
    - ``explicit``       – cast to int (1 / 0)
    - ``encoded_genre``  – label-encoded ``track_genre``
    - ``mood_score``     – mean of ``valence`` and ``energy``
    - ``hit``            – 1 if ``popularity`` ≥ 0.65, else 0

    Parameters
    ----------
    df:
        Cleaned DataFrame from :func:`clean_data`.

    Returns
    -------
    Tuple[pd.DataFrame, LabelEncoder]
        The enriched DataFrame and the fitted :class:`LabelEncoder`
        (saved so the Streamlit app can decode genres).
    """
    df = df.copy()

    df["popularity"] = df["popularity"] / 100
    df["duration_mins"] = df["duration_ms"] / 60_000
    df["explicit"] = df["explicit"].astype(int)

    le = LabelEncoder()
    df["encoded_genre"] = le.fit_transform(df["track_genre"])

    df["mood_score"] = (df["valence"] + df["energy"]) / 2
    df["hit"] = (df["popularity"] >= 0.65).astype(int)

    print("Feature engineering complete.")
    return df, le


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def scale_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Min-max scale *features* on a copy of *df*.

    Parameters
    ----------
    df:
        Engineered DataFrame from :func:`engineer_features`.
    features:
        Column names to scale. Defaults to :data:`AUDIO_FEATURES`.

    Returns
    -------
    Tuple[pd.DataFrame, MinMaxScaler]
        A scaled copy of *df* and the fitted scaler.
    """
    if features is None:
        features = AUDIO_FEATURES

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    print(f"Scaled {len(features)} features.")
    return df_scaled, scaler


def build_rec_matrix(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, MinMaxScaler]:
    """Build the recommendation feature matrix used by cosine similarity.

    Parameters
    ----------
    df:
        Engineered (unscaled) DataFrame.
    features:
        Columns to include. Defaults to :data:`REC_FEATURES`.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, MinMaxScaler]
        - *df_rec*: subset DataFrame with metadata columns appended.
        - *X_rec*: scaled numpy array (one row per track).
        - *rec_scaler*: the fitted :class:`MinMaxScaler`.
    """
    if features is None:
        features = REC_FEATURES

    df_rec = (
        df[features + ["track_name", "artists", "track_genre", "popularity"]]
        .reset_index(drop=True)
    )
    rec_scaler = MinMaxScaler()
    X_rec = rec_scaler.fit_transform(df_rec[features])
    print(f"Recommendation matrix built: {X_rec.shape}")
    return df_rec, X_rec, rec_scaler


def build_cluster_matrix(
    df: pd.DataFrame,
    n_sample: int = 20_000,
    features: list[str] | None = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, MinMaxScaler]:
    """Sample and scale the cluster feature matrix.

    Parameters
    ----------
    df:
        Engineered DataFrame.
    n_sample:
        Number of songs to sample (set to ``len(df)`` to use all).
    features:
        Columns to include. Defaults to :data:`CLUSTER_FEATURES`.
    random_state:
        Random seed for reproducible sampling.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, MinMaxScaler]
        - *df_cluster*: sampled DataFrame (no cluster column yet).
        - *X_cluster*: scaled numpy array.
        - *cluster_scaler*: the fitted :class:`MinMaxScaler`.
    """
    if features is None:
        features = CLUSTER_FEATURES

    df_cluster = df.sample(min(n_sample, len(df)), random_state=random_state).reset_index(drop=True)
    cluster_scaler = MinMaxScaler()
    X_cluster = cluster_scaler.fit_transform(df_cluster[features])
    print(f"Cluster matrix built: {X_cluster.shape}")
    return df_cluster, X_cluster, cluster_scaler


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_cleaned_data(df: pd.DataFrame, path: str = "data/cleaned_spotify.csv") -> None:
    """Write *df* to *path* as CSV.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Destination path (parent directories are created if needed).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned data saved to '{path}'")


def save_artefacts(
    scaler: MinMaxScaler,
    feature_columns: list[str],
    models_dir: str = "models",
) -> None:
    """Persist the scaler and feature-column list used by hit prediction.

    Parameters
    ----------
    scaler:
        The :class:`MinMaxScaler` fitted on :data:`AUDIO_FEATURES`.
    feature_columns:
        Ordered list of feature names (must match training order).
    models_dir:
        Directory where ``.pkl`` files are written.
    """
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(models_dir, "feature_columns.pkl"))
    print(f"Scaler and feature columns saved to '{models_dir}/'")