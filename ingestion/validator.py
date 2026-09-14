# ingestion/validator.py

import logging

import numpy as np
import pandas as pd

from configs.settings import REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


def _quarantine_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that are structurally unusable: missing IDs, bad ranges, bad Churn."""
    n = len(df)
    mask = pd.Series(False, index=df.index)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            # Missing column entirely -> everything is quarantined
            return pd.Series(True, index=df.index)
    # customerID missing
    mask = mask | df["customerID"].isna() | (df["customerID"].astype(str).str.strip() == "")
    # tenure must be numeric >= 0
    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    mask = mask | tenure.isna() | (tenure < 0)
    # MonthlyCharges numeric >= 0
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    mask = mask | monthly.isna() | (monthly < 0)
    # Churn domain
    mask = mask | ~df["Churn"].isin(["Yes", "No"])
    return mask


def validate_data(df: pd.DataFrame, return_report: bool = False):
    """
    Permissive validation: quarantine bad rows, never abort on them.

    - Schema check on REQUIRED_COLUMNS
    - TotalCharges stripped -> to_numeric(coerce) first, then typed imputation
      (numeric median + is_totalcharges_missing flag; categorical mode)
    - drop_duplicates(subset=["customerID"], keep="first")
    - Returns clean_df for backward compat, or (clean, quarantine, report)
      when return_report=True.
    """
    logger.info("Starting data validation...")
    df = df.copy()
    report: dict = {"n_input": int(len(df))}

    # 1. Schema check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning("Missing required columns: %s", missing_cols)
        report["missing_columns"] = missing_cols

    # 2. Normalize TotalCharges early (strip -> coerce) so imputation sees truth
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )
        report["totalcharges_coerced_nulls"] = int(df["TotalCharges"].isna().sum())

    # 3. Split quarantine (structural) vs cleanable
    qmask = _quarantine_mask(df)
    quarantine_df = df[qmask].copy()
    clean_df = df[~qmask].copy()
    report["n_quarantined"] = int(len(quarantine_df))

    # 4. Deduplicate clean on customerID
    before = len(clean_df)
    if "customerID" in clean_df.columns:
        clean_df = clean_df.drop_duplicates(subset=["customerID"], keep="first")
    else:
        clean_df = clean_df.drop_duplicates(keep="first")
    report["n_deduped"] = int(before - len(clean_df))

    # 5. Typed imputation on clean (no ffill)
    if "TotalCharges" in clean_df.columns:
        clean_df["is_totalcharges_missing"] = clean_df["TotalCharges"].isna().astype(int)
        median_val = clean_df["TotalCharges"].median()
        if pd.isna(median_val):
            median_val = 0.0
        clean_df["TotalCharges"] = clean_df["TotalCharges"].fillna(median_val)
    for col in clean_df.columns:
        if clean_df[col].dtype == object or str(clean_df[col].dtype) == "category":
            if clean_df[col].isna().any():
                mode = clean_df[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else "Unknown"
                clean_df[col] = clean_df[col].fillna(fill)
        elif np.issubdtype(clean_df[col].dtype, np.number):
            if clean_df[col].isna().any() and col != "TotalCharges":
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # Coerce core numerics
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    report["n_output"] = int(len(clean_df))
    logger.info(
        "Data validation complete. in=%d quarantined=%d out=%d",
        report["n_input"],
        report["n_quarantined"],
        report["n_output"],
    )
    if return_report:
        return clean_df, quarantine_df, report
    return clean_df
