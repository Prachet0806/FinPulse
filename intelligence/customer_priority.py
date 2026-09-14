# intelligence/customer_priority.py

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_priority(df):
    """
    Priority score = churn risk x CLV (fail-closed: NaN -> 0).
    """
    df = df.copy()
    churn = pd.to_numeric(df.get("churn_probability"), errors="coerce").fillna(0.0)
    clv = pd.to_numeric(df.get("estimated_clv"), errors="coerce").fillna(0.0)
    df["priority_score"] = churn * clv
    if bool((df["priority_score"].skew(skipna=True) or 0) > 5):
        logger.info("priority_score highly right-skewed; kept unnormalized for dashboard stability")
    return df
