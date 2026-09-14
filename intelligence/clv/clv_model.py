# intelligence/clv/clv_model.py

import logging

import numpy as np
import pandas as pd

from configs.settings import CLV_MIN_DENOM, DISCOUNT_RATE

logger = logging.getLogger(__name__)


def estimate_clv(df):
    """
    Estimate CLV using the financial retention formula:
    CLV = (ARPU * Retention) / (1 + Discount - Retention)

    Fail-closed: missing/NaN churn_probability -> retention 0 (CLV ~ 0).
    Infinities quarantined to 0 + flag.
    """
    df = df.copy()
    logger.info("Estimating advanced CLV...")

    # Fail-closed: unknown churn -> max risk -> retention 0
    churn_prob = pd.to_numeric(df.get("churn_probability", 0), errors="coerce").fillna(1.0)
    retention = (1 - churn_prob).clip(0, 1)

    # ARPU from behavioral features
    arpu = pd.to_numeric(df.get("avg_revenue_per_month"), errors="coerce").fillna(0.0)

    denom = (1 + DISCOUNT_RATE - retention).clip(lower=CLV_MIN_DENOM)
    with np.errstate(divide="ignore", invalid="ignore"):
        clv = (arpu * retention) / denom

    bad = ~np.isfinite(clv.to_numpy(dtype=float))
    df["is_clv_quarantined"] = pd.Series(bad, index=df.index).astype(int)
    if int(bad.sum()) > 0:
        logger.warning("Quarantined %d non-finite CLV rows to 0", int(bad.sum()))
    clv = pd.Series(np.where(bad, 0.0, clv), index=df.index, dtype=float)

    df["estimated_clv"] = clv
    return df
