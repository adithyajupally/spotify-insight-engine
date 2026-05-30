"""
src/prediction.py
-----------------
Hit-song classification: training, evaluation, saving, and inference.
All functions accept explicit arguments (no global state).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_hit_data(
    df: pd.DataFrame,
    features: list[str],
    target: str = "hit",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train / test sets for hit prediction.

    Parameters
    ----------
    df:
        Engineered DataFrame containing *features* and *target* columns.
    features:
        List of predictor column names.
    target:
        Binary target column name (default ``'hit'``).
    test_size:
        Fraction of data reserved for testing.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    Tuple[X_train, X_test, y_train, y_test]
    """
    X = df[features]
    y = df[target]

    print(f"Hit songs: {y.sum():,}  |  Non-hit songs: {(~y.astype(bool)).sum():,}")
    print(f"Class ratio: {y.mean() * 100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.7,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class with SMOTE.

    Parameters
    ----------
    X_train:
        Training features before resampling.
    y_train:
        Training labels before resampling.
    sampling_strategy:
        Ratio of minority to majority class after resampling.
    random_state:
        Random seed.

    Returns
    -------
    Tuple[X_resampled, y_resampled]
        Arrays ready to be passed to ``model.fit``.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Hit: {y_res.sum():,}  |  Non-hit: {(~y_res.astype(bool)).sum():,}")
    print(f"New class ratio: {y_res.mean() * 100:.2f}%")
    return X_res, y_res


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def get_default_models(random_state: int = 42) -> Dict[str, Any]:
    """Return a dict of {name: unfitted_model} for benchmarking.

    Parameters
    ----------
    random_state:
        Seed passed to every model that accepts one.

    Returns
    -------
    Dict[str, estimator]
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=random_state),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit every model in *models* and return a sorted results DataFrame.

    Parameters
    ----------
    models:
        Dict of {name: unfitted_estimator} (e.g. from :func:`get_default_models`).
    X_train:
        Training features (post-SMOTE array).
    X_test:
        Test features.
    y_train:
        Training labels (post-SMOTE array).
    y_test:
        Test labels.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, fitted_estimator]]
        - *results_df*: one row per model, sorted by ROC AUC descending.
        - *fitted_models*: the same dict with models now fitted.
    """
    records: List[Dict] = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        records.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "F1 Score": round(f1_score(y_test, y_pred), 4),
                "ROC AUC": round(roc_auc_score(y_test, y_prob), 4),
            }
        )

    results_df = pd.DataFrame(records).sort_values(by="ROC AUC", ascending=False).reset_index(drop=True)
    return results_df, models


def get_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> str:
    """Return the full classification report string for *model*.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible classifier.
    X_test, y_test:
        Held-out evaluation data.
    model_name:
        Label used in the printed header.

    Returns
    -------
    str
        The text of :func:`sklearn.metrics.classification_report`.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-Hit", "Hit"])
    print(f"Classification Report — {model_name}:\n{report}")
    return report


def get_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Return a labelled confusion-matrix DataFrame.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible classifier.
    X_test, y_test:
        Held-out evaluation data.

    Returns
    -------
    pd.DataFrame
        2×2 confusion matrix with labelled rows and columns.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual Non-Hit", "Actual Hit"],
        columns=["Predicted Non-Hit", "Predicted Hit"],
    )


def get_feature_importance(
    model: RandomForestClassifier,
    features: list[str],
) -> pd.DataFrame:
    """Return feature importances as a DataFrame sorted descending.

    Parameters
    ----------
    model:
        A fitted tree-based model with ``feature_importances_``.
    features:
        Ordered list of feature names used during training.

    Returns
    -------
    pd.DataFrame
        Columns: ``['Importance']``, indexed by feature name.
    """
    return (
        pd.DataFrame(
            model.feature_importances_,
            index=features,
            columns=["Importance"],
        )
        .sort_values("Importance", ascending=False)
    )


def predict_with_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.3,
) -> str:
    """Evaluate *model* with a custom probability threshold.

    Parameters
    ----------
    model:
        A fitted classifier.
    X_test, y_test:
        Held-out evaluation data.
    threshold:
        Decision boundary probability (default 0.3 boosts hit recall).

    Returns
    -------
    str
        Classification report text.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_test, y_pred, target_names=["Non-Hit", "Hit"])
    print(f"Classification Report (threshold={threshold}):\n{report}")
    return report


def train_balanced_rf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, str]:
    """Train a class-weight-balanced Random Forest and print its report.

    Parameters
    ----------
    X_train, y_train:
        Raw (pre-SMOTE) training split.
    X_test, y_test:
        Test split.
    n_estimators:
        Number of trees.
    random_state:
        Random seed.

    Returns
    -------
    Tuple[fitted_model, report_string]
    """
    rf_balanced = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
    )
    rf_balanced.fit(X_train, y_train)
    report = get_classification_report(rf_balanced, X_test, y_test, "Balanced Random Forest")
    return rf_balanced, report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: Any, path: str = "models/hit_predictor.pkl") -> None:
    """Serialize *model* with joblib.

    Parameters
    ----------
    model:
        Any fitted sklearn-compatible estimator.
    path:
        Destination ``.pkl`` path (parent dirs are created).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to '{path}'")


# ---------------------------------------------------------------------------
# Inference (used by Streamlit app)
# ---------------------------------------------------------------------------


def predict_hit_probability(
    song_features: Dict[str, float],
    model_path: str = "models/hit_predictor.pkl",
    scaler_path: str = "models/scaler.pkl",
    feature_columns_path: str = "models/feature_columns.pkl",
) -> float:
    """Predict the hit probability for a single song.

    Loads the artefacts from disk on every call so the function is
    stateless and safe to use inside Streamlit.

    Parameters
    ----------
    song_features:
        Dict mapping feature name → raw (unscaled) value.
    model_path, scaler_path, feature_columns_path:
        Paths to the saved artefacts produced during training.

    Returns
    -------
    float
        Hit probability in [0, 1].
    """
    feature_order: list[str] = joblib.load(feature_columns_path)
    song_vector = np.array([song_features[f] for f in feature_order]).reshape(1, -1)

    scaler = joblib.load(scaler_path)
    song_vector_scaled = scaler.transform(song_vector)

    model = joblib.load(model_path)
    return float(model.predict_proba(song_vector_scaled)[0][1])