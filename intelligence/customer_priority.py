# intelligence/customer_priority.py

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_priority(df):
    """
    Priority score = churn risk x CLV (fail-closed: NaN -> 0).
    """
    df = df.copy()
    churn = pd.to_numeric(
        df.get("churn_probability", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    clv = pd.to_numeric(
        df.get("estimated_clv", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    churn = churn.where(churn.between(0, 1) & np.isfinite(churn), 0.0)
    clv = clv.where((clv >= 0) & np.isfinite(clv), 0.0)
    df["priority_score"] = churn * clv
    if bool((df["priority_score"].skew(skipna=True) or 0) > 5):
        logger.info("priority_score highly right-skewed; kept unnormalized for dashboard stability")
    return df
