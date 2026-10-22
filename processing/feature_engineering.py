# processing/feature_engineering.py

import json
import logging
import os

import numpy as np
import pandas as pd

from configs.settings import (
    ENGAGEMENT_WTS,
    FINTECH_SCALE,
    MODELS_DIR,
    TENURE_BINS,
)

logger = logging.getLogger(__name__)

ENCODER_COLUMNS_PATH = os.path.join(MODELS_DIR, "encoder_columns.json")

_CATEGORICAL_COLS = [
    "account_plan",
    "payment_behavior",
    "primary_product",
    "tenure_segment",
]


def fintech_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map telecom columns to fintech equivalents.
    Idempotent: skips FINTECH_SCALE multiplier if already applied.
    """
    df = df.copy()
    # Idempotency guard
    if "monthly_spend" in df.columns and "lifetime_value" in df.columns:
        logger.debug("fintech columns already present; skipping rename/scale")
        return df

    rename_map = {
        "tenure": "account_tenure_months",
        "MonthlyCharges": "monthly_spend",
        "TotalCharges": "lifetime_value",
        "Contract": "account_plan",
        "PaymentMethod": "payment_behavior",
        "InternetService": "primary_product",
        "TechSupport": "priority_support",
        "OnlineSecurity": "fraud_protection",
        "PaperlessBilling": "digital_adoption",
    }
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        # Permissive: quarantine via warning, continue with what exists
        logger.warning("fintech_rename: missing source columns %s", missing)
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Scale proxy data to realistic fintech economics (skip if absent)
    for col in ("monthly_spend", "lifetime_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * FINTECH_SCALE
        else:
            logger.warning("fintech_rename: expected column %s absent after rename", col)
    return df


def create_behavioral_features(
    df: pd.DataFrame, engagement_bounds: tuple[float, float] | None = None
) -> pd.DataFrame:
    """
    Create customer intelligence features (fail-closed on tenure==0).
    """
    df = df.copy()

    tenure = pd.to_numeric(df.get("account_tenure_months"), errors="coerce")
    lifetime = pd.to_numeric(df.get("lifetime_value"), errors="coerce")

    # Fail-closed: tenure==0 / missing -> NaN + flag (not silent 0)
    is_zero = (tenure == 0) | tenure.isna()
    df["is_zero_tenure"] = is_zero.astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = lifetime / tenure.mask(is_zero)
    # tenure==0 -> NaN (flagged); infinities -> NaN (quarantine downstream)
    avg = avg.replace([np.inf, -np.inf], np.nan)
    df["avg_revenue_per_month"] = avg

    # Engagement score: weighted combo, min-max normalized to [0, 1]
    w1, w2 = ENGAGEMENT_WTS
    monthly = pd.to_numeric(df.get("monthly_spend"), errors="coerce").fillna(0.0)
    avg_fill = df["avg_revenue_per_month"].fillna(0.0)
    raw = monthly * w1 + avg_fill * w2
    lo, hi = engagement_bounds if engagement_bounds is not None else (raw.min(), raw.max())
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        df["engagement_score"] = ((raw - lo) / (hi - lo)).clip(0, 1)
    else:
        df["engagement_score"] = 0.0

    # Tenure category (bins from settings + inf so >72 never NaN)
    bins = list(TENURE_BINS) + [float("inf")]
    df["tenure_segment"] = pd.cut(
        tenure,
        bins=bins,
        labels=["new", "early", "mid", "loyal", "veteran"],
        include_lowest=True,
    )
    return df


def build_features(df: pd.DataFrame, save_encoder: bool = True) -> pd.DataFrame:
    """
    Legacy standalone feature builder. The full pipeline fits and applies
    FeatureTransformer after splitting labeled records.
    """
    logger.info("Applying fintech domain transformation...")

    df = fintech_rename_columns(df)
    df = create_behavioral_features(df)

    existing_cats = [c for c in _CATEGORICAL_COLS if c in df.columns]
    # tenure_segment keeps all levels (drop_first=False) so the
    # fail-closed eligibility gate `tenure_segment_new` always exists.
    # The reference "new" level would otherwise be dropped, silently
    # disqualifying every cashback_offer candidate.
    other_cats = [c for c in existing_cats if c != "tenure_segment"]
    # Keep every level. A batch-local dropped baseline can silently change the
    # meaning of a stored model's zero-valued dummy columns.
    df = pd.get_dummies(df, columns=other_cats, drop_first=False)
    if "tenure_segment" in existing_cats:
        df = pd.get_dummies(df, columns=["tenure_segment"], drop_first=False)

    if save_encoder:
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(ENCODER_COLUMNS_PATH, "w", encoding="utf-8") as f:
                json.dump(list(df.columns), f)
        except OSError:
            logger.warning("Could not persist encoder columns", exc_info=True)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def align_inference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex inference frame to persisted training columns (unseen -> 0)."""
    try:
        with open(ENCODER_COLUMNS_PATH, encoding="utf-8") as f:
            cols = json.load(f)
    except (OSError, ValueError):
        logger.warning("No persisted encoder columns; skipping alignment")
        return df
    return df.reindex(columns=cols, fill_value=0)


class FeatureTransformer:
    """Fitted, batch-independent feature transformation for model bundles."""

    def __init__(self):
        self.total_median: float | None = None
        self.engagement_bounds: tuple[float, float] | None = None
        self.categories: dict[str, list[str]] = {}
        self.numeric_medians: dict[str, float] = {}

    def _base(self, raw: pd.DataFrame) -> pd.DataFrame:
        if self.total_median is None:
            raise ValueError("FeatureTransformer must be fitted before transform")
        df = fintech_rename_columns(raw)
        total = pd.to_numeric(df["lifetime_value"], errors="coerce")
        df["is_totalcharges_missing"] = total.isna().astype(int)
        df["lifetime_value"] = total.fillna(self.total_median)
        return create_behavioral_features(df, self.engagement_bounds)

    def fit(self, raw: pd.DataFrame):
        df = fintech_rename_columns(raw)
        total = pd.to_numeric(df["lifetime_value"], errors="coerce")
        median = total.median()
        self.total_median = float(median) if pd.notna(median) else 0.0
        df["lifetime_value"] = total.fillna(self.total_median)
        base = create_behavioral_features(df)
        avg = base["avg_revenue_per_month"].fillna(0.0)
        monthly = pd.to_numeric(base["monthly_spend"], errors="coerce").fillna(0.0)
        raw_engagement = monthly * ENGAGEMENT_WTS[0] + avg * ENGAGEMENT_WTS[1]
        self.engagement_bounds = (float(raw_engagement.min()), float(raw_engagement.max()))
        for col in _CATEGORICAL_COLS:
            if col == "tenure_segment":
                self.categories[col] = ["new", "early", "mid", "loyal", "veteran", "Unknown"]
            elif col in base.columns:
                self.categories[col] = sorted(base[col].astype("string").fillna("Unknown").unique().tolist())
            else:
                self.categories[col] = ["Unknown"]
        encoded = self._encode(self._base(raw))
        self.numeric_medians = {}
        for col in encoded.select_dtypes(include=["number", "bool"]).columns:
            median = pd.to_numeric(encoded[col], errors="coerce").median()
            self.numeric_medians[col] = float(median) if pd.notna(median) else 0.0
        return self

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, categories in self.categories.items():
            values = (
                df[col].astype("string").fillna("Unknown")
                if col in df.columns
                else pd.Series("Unknown", index=df.index, dtype="string")
            )
            df[f"is_{col}_unseen"] = (~values.isin(categories)).astype("int8")
            for category in categories:
                df[f"{col}_{category}"] = (values == category).astype("int8")
        return df.drop(columns=[c for c in _CATEGORICAL_COLS if c in df.columns])

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if self.engagement_bounds is None:
            raise ValueError("FeatureTransformer must be fitted before transform")
        df = self._encode(self._base(raw))
        for col, median in self.numeric_medians.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)
            else:
                df[col] = median
        return df
