import numpy as np
import pandas as pd

from decision_engine.eligibility import apply_eligibility_mask


def test_eligibility_failclosed_nan(featured_frame):
    df = featured_frame
    base = pd.Series([100.0] * len(df), index=df.index)
    # Row 5 has NaN churn -> ineligible for churn-gated rules
    assert apply_eligibility_mask(df, "credit_limit_increase", base).iloc[5] == -np.inf
    # cashback_offer gates on tenure_segment_new only (row 5 flag valid -> eligible)
    assert apply_eligibility_mask(df, "cashback_offer", base).iloc[5] == 100.0
    # loan_offer gates on CLV only (row 5 CLV valid -> eligible)
    assert apply_eligibility_mask(df, "loan_offer", base).iloc[5] == 100.0


def test_eligibility_rules():
    df = pd.DataFrame({
        "account_tenure_months": [2, 24, 24, 24],
        "churn_probability": [0.1, 0.9, 0.1, 0.1],
        "estimated_clv": [5000.0, 5000.0, 100.0, 5000.0],
        "tenure_segment_new": [0, 0, 0, 1],
    })
    base = pd.Series([10.0] * 4)
    assert apply_eligibility_mask(df, "credit_limit_increase", base).iloc[0] == -np.inf
    assert apply_eligibility_mask(df, "credit_limit_increase", base).iloc[1] == -np.inf
    assert apply_eligibility_mask(df, "loan_offer", base).iloc[2] == -np.inf
    assert apply_eligibility_mask(df, "cashback_offer", base).iloc[3] == -np.inf
    # fee_waiver always eligible
    assert (apply_eligibility_mask(df, "fee_waiver", base) == 10.0).all()


def test_eligibility_missing_column_quarantines():
    df = pd.DataFrame({"account_tenure_months": [24], "churn_probability": [0.1]})
    base = pd.Series([10.0])
    # No tenure_segment_new -> cashback ineligible (not skip)
    assert apply_eligibility_mask(df, "cashback_offer", base).iloc[0] == -np.inf
