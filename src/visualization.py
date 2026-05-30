"""
src/visualization.py
--------------------
All plotting helpers used in EDA and model evaluation.
Every function accepts explicit DataFrame / array arguments so there
are no hidden global dependencies.
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Plot a correlation heatmap for all numeric columns in *df*.

    Parameters
    ----------
    df:
        DataFrame whose numeric columns are correlated.
    title:
        Figure title.
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.
    """
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_popularity_distribution(
    df: pd.DataFrame,
    bins: int = 20,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """Histogram + KDE of the ``popularity`` column.

    Parameters
    ----------
    df:
        DataFrame with a ``'popularity'`` column.
    bins:
        Number of histogram bins.
    figsize:
        Figure size.
    """
    plt.figure(figsize=figsize)
    sns.histplot(df["popularity"], bins=bins, kde=True)
    plt.title("Popularity Distribution")
    plt.xlabel("Popularity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_top_genres(
    df: pd.DataFrame,
    n: int = 20,
) -> None:
    """Interactive bar chart of the top-*n* genres by mean popularity.

    Parameters
    ----------
    df:
        DataFrame with ``'track_genre'`` and ``'popularity'`` columns.
    n:
        Number of top genres to display.
    """
    top_genres = (
        df.groupby("track_genre")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(n)
    )
    fig = px.bar(
        x=top_genres.index,
        y=top_genres.values,
        title=f"Top {n} Genres by Average Popularity",
        labels={"x": "Genre", "y": "Average Popularity"},
        color=top_genres.values,
        color_continuous_scale="Viridis",
    )
    fig.show()


def plot_audio_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    bins: int = 20,
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """Histogram grid for every audio feature in *features*.

    Parameters
    ----------
    df:
        DataFrame containing *features*.
    features:
        List of column names to plot.
    bins:
        Number of histogram bins per subplot.
    figsize:
        Overall figure size.
    """
    df[features].hist(bins=bins, figsize=figsize)
    plt.suptitle("Audio Feature Distributions", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_feature_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Correlation heatmap restricted to *features* + ``popularity``.

    Parameters
    ----------
    df:
        DataFrame containing *features* and ``'popularity'``.
    features:
        Audio feature column names.
    figsize:
        Figure size.
    """
    cols = features + ["popularity"]
    corr = df[cols].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Audio Features × Popularity Correlation")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Model evaluation helpers
# ---------------------------------------------------------------------------


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "ROC AUC",
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """Horizontal bar chart comparing models on *metric*.

    Parameters
    ----------
    results_df:
        DataFrame returned by :func:`prediction.train_and_evaluate`.
    metric:
        Column name to visualise (default ``'ROC AUC'``).
    figsize:
        Figure size.
    """
    sorted_df = results_df.sort_values(metric)
    plt.figure(figsize=figsize)
    plt.barh(sorted_df["Model"], sorted_df[metric], color="steelblue")
    plt.xlabel(metric)
    plt.title(f"Model Comparison — {metric}")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Horizontal bar chart of the top-*n* feature importances.

    Parameters
    ----------
    importance_df:
        DataFrame with a single ``'Importance'`` column, indexed by
        feature names (output of :func:`prediction.get_feature_importance`).
    top_n:
        Number of features to display.
    figsize:
        Figure size.
    """
    top = importance_df.head(top_n).sort_values("Importance")
    plt.figure(figsize=figsize)
    plt.barh(top.index, top["Importance"], color="teal")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()