# decision_engine/risk_adjustment.py

import logging

import pandas as pd

from configs.settings import RISK_SLOPES

logger = logging.getLogger(__name__)


def calculate_risk_penalty(
    df: pd.DataFrame,
    strategy: str,
    params: dict,
    use_clv_scaling: bool = False,
) -> pd.Series:
    """
    Calculate the risk penalty for a given strategy per customer.

    - Fail-closed: NaN churn -> max penalty (treated as churn=1.0).
    - Default flat mode preserves historical behavior; both flat and
      CLV-scaled values are logged when scaling is enabled.
    """
    base_penalty = params.get("risk_penalty", 0.0)
    churn = pd.to_numeric(df.get("churn_probability"), errors="coerce").fillna(1.0)
    if bool(churn.isna().any()):
        logger.warning("Risk quarantine: NaN churn after fill for %s", strategy)

    slope = RISK_SLOPES.get(strategy)
    if strategy == "credit_limit_increase" and slope is not None:
        flat = base_penalty + (churn * slope)
    elif strategy == "loan_offer" and slope is not None:
        flat = base_penalty + (churn * slope)
    else:
        # Default is the static base penalty
        return pd.Series(base_penalty, index=df.index)

    if not use_clv_scaling:
        return flat

    clv = pd.to_numeric(df.get("estimated_clv"), errors="coerce").fillna(0.0)
    scale = (clv / 1000.0).clip(lower=0)
    scaled = base_penalty + (churn * slope * (1 + scale))
    logger.info(
        "Risk %s: flat_mean=%.2f scaled_mean=%.2f",
        strategy,
        float(flat.mean()),
        float(scaled.mean()),
    )
    return scaled
