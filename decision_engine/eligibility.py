# decision_engine/eligibility.py

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_eligibility_mask(df: pd.DataFrame, strategy: str, net_value_series: pd.Series) -> pd.Series:
    """
    Apply business rules to qualify/disqualify customers (fail-closed).

    - NaN/missing in any gating column -> ineligible (-inf).
    - Missing tenure_segment_new -> ineligible for cashback (not skip).
    - Missing gating columns entirely -> quarantine warning + ineligible.
    - fee_waiver is always eligible by design (no gating rule).
    """
    adjusted_series = net_value_series.copy()

    def _col(name: str) -> pd.Series:
        if name not in df.columns:
            logger.warning("Eligibility quarantine: missing column %s for %s", name, strategy)
            return pd.Series(np.nan, index=df.index)
        return df[name]

    # 1. Credit Limit Increase Rule: Requires tenure > 12 months & low churn risk
    if strategy == "credit_limit_increase":
        tenure = pd.to_numeric(_col("account_tenure_months"), errors="coerce")
        churn = pd.to_numeric(_col("churn_probability"), errors="coerce")
        ineligible_mask = (
            (tenure <= 12) | (churn > 0.6) | tenure.isna() | churn.isna()
        )
        adjusted_series[ineligible_mask] = -np.inf

    # 2. Loan Offer Rule: Only eligible if Estimated CLV is high enough to justify
    elif strategy == "loan_offer":
        clv = pd.to_numeric(_col("estimated_clv"), errors="coerce")
        ineligible_mask = (clv < 500) | clv.isna()
        adjusted_series[ineligible_mask] = -np.inf

    # 3. Cashback Offer Rule: Exclude purely 'new' users to avoid gamification
    elif strategy == "cashback_offer":
        if "tenure_segment_new" not in df.columns:
            logger.warning(
                "Eligibility quarantine: tenure_segment_new missing; "
                "cashback_offer ineligible for all rows"
            )
            adjusted_series[:] = -np.inf
        else:
            flag = pd.to_numeric(df["tenure_segment_new"], errors="coerce")
            ineligible_mask = (flag == 1) | flag.isna()
            adjusted_series[ineligible_mask] = -np.inf

    # fee_waiver / no_action: always eligible (documented)

    return adjusted_series
