# processing/feature_engineering.py

import pandas as pd
import numpy as np
import logging
from configs.settings import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def fintech_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map telecom columns to fintech equivalents.
    Applies a 10x multiplier to simulate standard banking/fintech balances.
    """

    rename_map = {
        "tenure": "account_tenure_months",
        "MonthlyCharges": "monthly_spend",
        "TotalCharges": "lifetime_value",
        "Contract": "account_plan",
        "PaymentMethod": "payment_behavior",
        "InternetService": "primary_product",
        "TechSupport": "priority_support",
        "OnlineSecurity": "fraud_protection",
        "PaperlessBilling": "digital_adoption"
    }

    df = df.rename(columns=rename_map)
    
    # Scale proxy data to realistic Fintech Economics
    df["monthly_spend"] = df["monthly_spend"] * 10
    df["lifetime_value"] = df["lifetime_value"] * 10
    
    return df


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer intelligence features.
    """

    # Revenue-based feature
    # Handle division by zero and resulting infs
    df["avg_revenue_per_month"] = (
        df["lifetime_value"] / df["account_tenure_months"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Engagement score proxy
    df["engagement_score"] = (
        df["monthly_spend"] * 0.5 +
        df["avg_revenue_per_month"] * 0.5
    )

    # Tenure category
    df["tenure_segment"] = pd.cut(
        df["account_tenure_months"],
        bins=[0, 12, 24, 48, 72],
        labels=["new", "early", "mid", "loyal"],
        include_lowest=True
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    """

    logger.info("Applying fintech domain transformation...")

    df = fintech_rename_columns(df)
    df = create_behavioral_features(df)

    # Categorical encoding (High level)
    categorical_cols = [
        "account_plan", 
        "payment_behavior", 
        "primary_product",
        "tenure_segment"
    ]
    
    # Ensure columns exist before encoding
    existing_cats = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df