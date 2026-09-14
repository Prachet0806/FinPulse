# intelligence/churn/churn_model.py

import json
import logging
import os
import tempfile

import joblib
import pandas as pd
from configs.settings import MODELS_DIR, RANDOM_SEED, TEST_SIZE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Feedback-loop columns that must never become model features
_FEEDBACK_COLS = {
    "churn_probability",
    "estimated_clv",
    "priority_score",
    "retention_gain",
    "recommended_strategy",
}

_ID_COLS = {"customerID", "segment_id", "segment_label"}

# Single shared constant: train and score must use identical columns
FEATURE_COLS: list | None = None
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "churn_feature_columns.json")
MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "churn_metrics.json")


def _build_feature_frame(df: pd.DataFrame):
    """Numeric feature frame minus target/IDs/feedback; returns (X, target_col)."""
    if "Churn" not in df.columns:
        logger.warning("'Churn' column not found, checking for case variation")
        target_cols = [c for c in df.columns if c.lower() == "churn"]
        if not target_cols:
            raise ValueError("Churn column not found in dataset")
        target_col = target_cols[0]
    else:
        target_col = "Churn"

    X = df.select_dtypes(include=["int64", "float64", "uint8", "int32", "bool"])
    drop = {target_col} | _FEEDBACK_COLS | _ID_COLS
    X = X.drop(columns=[c for c in drop if c in X.columns])
    # Align bools to ints for sklearn stability
    for c in X.columns:
        if str(X[c].dtype) == "bool":
            X[c] = X[c].astype("int64")
    return X, target_col


def train_churn_model(df: pd.DataFrame, save_model=True):
    """
    Train churn prediction model and optionally save it.
    """
    global FEATURE_COLS

    logger.info("Training churn model...")

    X, target_col = _build_feature_frame(df)
    y = df[target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    # Quarantine-style: drop rows with unmappable target
    valid = y.notna()
    if not bool(valid.all()):
        logger.warning("Dropping %d rows with invalid Churn values", int((~valid).sum()))
        X, y = X[valid], y[valid].astype(int)

    FEATURE_COLS = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        max_depth=12,
        min_samples_leaf=5,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(model.score(X_test, y_test)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_features": int(X.shape[1]),
        "feature_columns": FEATURE_COLS,
    }
    logger.info(
        "Churn model complete. acc=%.2f prec=%.2f rec=%.2f f1=%.2f auc=%.2f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )

    if save_model:
        os.makedirs(MODELS_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=MODELS_DIR, suffix=".tmp")
        os.close(fd)
        try:
            joblib.dump(model, tmp)
            os.replace(tmp, MODEL_PATH)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
            json.dump(FEATURE_COLS, f)
        # Attach to model for in-process scoring parity
        try:
            model.feature_names_in_ = FEATURE_COLS  # type: ignore[attr-defined]
        except Exception:
            pass
        logger.info("Model saved to %s", MODEL_PATH)

    return model


def load_churn_model():
    """Load the churn model from disk (restores FEATURE_COLS)."""
    global FEATURE_COLS
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        if os.path.exists(FEATURE_COLS_PATH):
            try:
                with open(FEATURE_COLS_PATH, encoding="utf-8") as f:
                    FEATURE_COLS = json.load(f)
            except (OSError, ValueError):
                pass
        if FEATURE_COLS is None and hasattr(model, "feature_names_in_"):
            try:
                FEATURE_COLS = list(model.feature_names_in_)
            except TypeError:
                pass
        return model
    return None
