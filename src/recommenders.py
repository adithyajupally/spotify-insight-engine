"""
src/recommenders.py
-------------------
Content-based, genre, artist, mood, cluster-based, and hybrid
recommendation functions. Every function is stateless: data and
feature matrices are passed in explicitly so there are no module-level
globals and the functions can be called from anywhere.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Content-based (cosine similarity over full dataset)
# ---------------------------------------------------------------------------


def normal_recommendation(
    song_name: str,
    df_rec: pd.DataFrame,
    X_rec: np.ndarray,
    top_n: int = 10,
    min_popularity: float = 0.65,
) -> Optional[pd.DataFrame]:
    """Recommend songs by cosine similarity of audio features.

    Parameters
    ----------
    song_name:
        Track name to look up (case-insensitive).
    df_rec:
        DataFrame with columns ``[rec_features..., 'track_name',
        'artists', 'track_genre', 'popularity']``.
    X_rec:
        Scaled feature matrix corresponding to *df_rec* rows.
    top_n:
        Number of recommendations to return.
    min_popularity:
        Minimum normalised popularity score [0, 1] for a song to be
        included in results.

    Returns
    -------
    Optional[pd.DataFrame]
        Top-*n* recommendations, or ``None`` if the song was not found.
    """
    matches = df_rec[df_rec["track_name"].str.lower() == song_name.lower()]
    if matches.empty:
        print(f'Song "{song_name}" not found.')
        return None

    idx = matches.index[0]
    song_vector = X_rec[idx].reshape(1, -1)
    sim_scores = cosine_similarity(song_vector, X_rec)[0]

    temp_df = df_rec.copy()
    temp_df["similarity_score"] = sim_scores
    temp_df = temp_df[temp_df["similarity_score"] != 1]
    temp_df = temp_df[temp_df["popularity"] >= min_popularity]
    temp_df = temp_df.sort_values("similarity_score", ascending=False)
    temp_df = temp_df.drop_duplicates(subset="track_name")

    return (
        temp_df[["track_name", "artists", "track_genre", "popularity", "similarity_score"]]
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Artist look-up
# ---------------------------------------------------------------------------


def artist_songs(
    artist_name: str,
    df_rec: pd.DataFrame,
    top_n: int = 10,
) -> Optional[pd.DataFrame]:
    """Return the top-*n* most popular songs by *artist_name*.

    Parameters
    ----------
    artist_name:
        Artist to search for (case-insensitive substring match).
    df_rec:
        Recommendation DataFrame (see :func:`normal_recommendation`).
    top_n:
        Maximum number of tracks to return.

    Returns
    -------
    Optional[pd.DataFrame]
        Matching tracks sorted by popularity, or ``None`` if not found.
    """
    mask = df_rec["artists"].str.lower().str.contains(artist_name.lower(), na=False)
    found = df_rec[mask]

    if found.empty:
        print(f'No songs found for artist "{artist_name}".')
        return None

    return (
        found[["track_name", "artists", "track_genre", "popularity"]]
        .drop_duplicates(subset="track_name")
        .sort_values("popularity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Genre look-up
# ---------------------------------------------------------------------------


def recommend_genres(
    genre: str,
    df_rec: pd.DataFrame,
    top_n: int = 20,
) -> Optional[pd.DataFrame]:
    """Return the top-*n* most popular songs in *genre*.

    Parameters
    ----------
    genre:
        Genre string to match (case-insensitive exact match).
    df_rec:
        Recommendation DataFrame.
    top_n:
        Maximum number of tracks to return.

    Returns
    -------
    Optional[pd.DataFrame]
        Top tracks in the genre, or ``None`` if genre not found.
    """
    mask = df_rec["track_genre"].str.lower() == genre.lower()
    genre_songs = df_rec[mask]

    if genre_songs.empty:
        print(f'Genre "{genre}" not found.')
        return None

    return (
        genre_songs.drop_duplicates(subset="track_name")
        .sort_values("popularity", ascending=False)
        .head(top_n)[["track_name", "artists", "popularity"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Mood-based playlist
# ---------------------------------------------------------------------------

MOOD_RULES: Dict[str, Tuple[str, str, float, float]] = {
    # mood: (feature_hi, feature_lo, hi_thresh, lo_thresh) — see _apply_mood_rule
}

_MOOD_CONDITIONS = {
    "workout": lambda df: (df["energy"] > 0.7) & (df["tempo"] > 0.6),
    "study":   lambda df: (df["instrumentalness"] > 0.4) & (df["energy"] < 0.5),
    "relax":   lambda df: (df["acousticness"] > 0.6) & (df["energy"] < 0.4),
    "party":   lambda df: (df["danceability"] > 0.7) & (df["energy"] > 0.6),
    "happy":   lambda df: (df["valence"] > 0.7) & (df["energy"] > 0.5),
    "sad":     lambda df: (df["valence"] < 0.3) & (df["energy"] < 0.5),
}


def mood_based(
    mood_type: str,
    df: pd.DataFrame,
    top_n: int = 10,
) -> Optional[pd.DataFrame]:
    """Generate a mood-based playlist.

    Parameters
    ----------
    mood_type:
        One of ``'workout'``, ``'study'``, ``'relax'``, ``'party'``,
        ``'happy'``, ``'sad'``.
    df:
        Engineered DataFrame (must contain the scaled audio feature
        columns used in the conditions above).
    top_n:
        Number of songs to include.

    Returns
    -------
    Optional[pd.DataFrame]
        Top-*n* tracks matching the mood, or ``None`` if unrecognised.
    """
    if mood_type not in _MOOD_CONDITIONS:
        valid = list(_MOOD_CONDITIONS.keys())
        print(f'Mood "{mood_type}" not recognised. Valid options: {valid}')
        return None

    condition = _MOOD_CONDITIONS[mood_type](df)
    filtered = df[condition]

    return (
        filtered.sort_values("popularity", ascending=False)
        .drop_duplicates(subset="track_name")
        .head(top_n)[["track_name", "artists", "popularity"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Cluster-based recommendation
# ---------------------------------------------------------------------------


def cluster_recommendation(
    song_name: str,
    df_cluster: pd.DataFrame,
    top_n: int = 10,
    min_popularity: float = 0.65,
) -> Optional[pd.DataFrame]:
    """Recommend songs from within the same K-Means cluster.

    Parameters
    ----------
    song_name:
        Track name to look up (case-insensitive).
    df_cluster:
        DataFrame with a ``'cluster'`` column (output of
        :func:`clustering.fit_kmeans`).
    top_n:
        Number of recommendations to return.
    min_popularity:
        Minimum normalised popularity to include.

    Returns
    -------
    Optional[pd.DataFrame]
        Top cluster-mates, or ``None`` if song not found.
    """
    matches = df_cluster[df_cluster["track_name"].str.lower() == song_name.lower()]
    if matches.empty:
        print(f'Song "{song_name}" not found in cluster dataset.')
        return None

    cluster_label = matches["cluster"].values[0]
    cluster_songs = df_cluster[
        (df_cluster["cluster"] == cluster_label)
        & (df_cluster["track_name"].str.lower() != song_name.lower())
        & (df_cluster["popularity"] >= min_popularity)
    ]

    return (
        cluster_songs.drop_duplicates(subset="track_name")
        .sort_values("popularity", ascending=False)
        .head(top_n)[["track_name", "artists", "track_genre", "popularity"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Hybrid recommendation
# ---------------------------------------------------------------------------


def hybrid_recommendation(
    song_name: str,
    df_rec: pd.DataFrame,
    X_rec: np.ndarray,
    df_cluster: pd.DataFrame,
    top_n: int = 10,
    min_popularity: float = 0.65,
    w_popularity: float = 0.5,
    w_similarity: float = 0.3,
    w_cluster: float = 0.2,
) -> Optional[pd.DataFrame]:
    """Blend content-based and cluster-based recommendations.

    The final score for each candidate track is:

        score = w_popularity × popularity
                + w_similarity × cosine_sim
                + w_cluster × (1 if also in cluster recs, else 0)

    Parameters
    ----------
    song_name:
        Query track name.
    df_rec:
        Recommendation DataFrame (metadata + features).
    X_rec:
        Scaled feature matrix.
    df_cluster:
        Clustered DataFrame (must have ``'cluster'`` column).
    top_n:
        Number of final recommendations.
    min_popularity:
        Minimum popularity threshold.
    w_popularity, w_similarity, w_cluster:
        Weights for the three scoring components (should sum to 1).

    Returns
    -------
    Optional[pd.DataFrame]
        Hybrid recommendations with a ``'combined_score'`` column.
    """
    cosine_recs = normal_recommendation(song_name, df_rec, X_rec, top_n * 2, min_popularity)
    if cosine_recs is None:
        return None

    cluster_recs = cluster_recommendation(song_name, df_cluster, top_n * 2, min_popularity)
    if cluster_recs is None:
        return None

    merged = pd.merge(
        cosine_recs,
        cluster_recs,
        on=["track_name", "artists", "track_genre", "popularity"],
        how="outer",
    )
    merged["similarity_score"] = merged["similarity_score"].fillna(0)
    merged["combined_score"] = (
        w_popularity * merged["popularity"]
        + w_similarity * merged["similarity_score"]
        + w_cluster * (merged["similarity_score"] > 0).astype(int)
    )

    filtered = merged[merged["popularity"] >= min_popularity]
    return (
        filtered.drop_duplicates(subset="track_name")
        .sort_values("combined_score", ascending=False)
        .head(top_n)[["track_name", "artists", "track_genre", "popularity", "combined_score"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Full comparison helper
# ---------------------------------------------------------------------------


def compare_all_recommenders(
    song_name: str,
    df_rec: pd.DataFrame,
    X_rec: np.ndarray,
    df_cluster: pd.DataFrame,
    top_n: int = 5,
) -> None:
    """Print side-by-side results from all three recommenders.

    Parameters
    ----------
    song_name:
        Query track name.
    df_rec:
        Recommendation DataFrame.
    X_rec:
        Scaled feature matrix.
    df_cluster:
        Clustered DataFrame.
    top_n:
        Number of results per recommender.
    """
    print(f"Comparing recommendations for: {song_name}\n")

    nr = normal_recommendation(song_name, df_rec, X_rec, top_n)
    cr = cluster_recommendation(song_name, df_cluster, top_n)
    hr = hybrid_recommendation(song_name, df_rec, X_rec, df_cluster, top_n)

    print("Content-Based Recommendations:")
    print(nr.to_string(index=False) if nr is not None else "Not found.")

    print("\nCluster-Based Recommendations:")
    print(cr.to_string(index=False) if cr is not None else "Not found.")

    print("\nHybrid Recommendations:")
    print(hr.to_string(index=False) if hr is not None else "Not found.")