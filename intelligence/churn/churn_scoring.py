# intelligence/churn/churn_scoring.py

import json
import logging
import os

import pandas as pd

from configs.settings import MODELS_DIR

logger = logging.getLogger(__name__)


def _resolve_feature_cols(model, df: pd.DataFrame) -> list:
    """Shared FEATURE_COLS parity: train columns win; else model attr; else numeric fallback."""
    try:
        from intelligence.churn.churn_model import FEATURE_COLS
    except ImportError:
        FEATURE_COLS = None
    cols = list(FEATURE_COLS) if FEATURE_COLS else None
    if not cols and hasattr(model, "feature_names_in_"):
        try:
            cols = list(model.feature_names_in_)
        except TypeError:
            cols = None
    if not cols:
        path = os.path.join(MODELS_DIR, "churn_feature_columns.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    cols = json.load(f)
            except (OSError, ValueError):
                cols = None
    if not cols:
        logger.warning("No saved feature columns; falling back to numeric dtypes")
        X = df.select_dtypes(include=["int64", "float64", "uint8", "int32", "bool"])
        drop = {"Churn", "churn", "customerID", "segment_id", "segment_label",
                "churn_probability", "estimated_clv", "priority_score",
                "retention_gain", "recommended_strategy"}
        cols = [c for c in X.columns if c not in drop]
    return cols


def score_churn(model, df):
    """
    Generate churn probabilities with train/serve column parity.
    """
    df = df.copy()
    cols = _resolve_feature_cols(model, df)
    X = df.reindex(columns=cols, fill_value=0)
    # Coerce bools/non-numeric leftovers
    for c in X.columns:
        if str(X[c].dtype) == "bool":
            X[c] = X[c].astype("int64")
        elif X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    probs = model.predict_proba(X)[:, 1]

    df["churn_probability"] = probs
    return df
