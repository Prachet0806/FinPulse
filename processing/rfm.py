# processing/rfm.py

import pandas as pd


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simplified RFM metrics.

    Recency  -> proxy using tenure
    Frequency -> engagement proxy
    Monetary -> lifetime value
    """

    df["recency_score"] = df["account_tenure_months"]
    df["frequency_score"] = df["engagement_score"]
    df["monetary_score"] = df["lifetime_value"]

    return df