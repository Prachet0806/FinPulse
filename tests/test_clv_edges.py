import numpy as np
import pandas as pd

from intelligence.clv.clv_model import estimate_clv


def _frame(churn, avg=200.0):
    return pd.DataFrame({
        "churn_probability": churn,
        "avg_revenue_per_month": [avg] * len(churn),
    })


def test_clv_fail_closed_nan():
    out = estimate_clv(_frame([float("nan")]))
    assert out["estimated_clv"].iloc[0] == 0.0


def test_clv_retention_clipped():
    # retention > 1 impossible; clipped to 1 -> finite positive CLV
    out = estimate_clv(_frame([-0.5]))
    assert np.isfinite(out["estimated_clv"].iloc[0])
    assert out["estimated_clv"].iloc[0] > 0


def test_clv_infinities_quarantined():
    out = estimate_clv(_frame([0.0], avg=float("inf")))
    assert out["estimated_clv"].iloc[0] == 0.0
    assert out["is_clv_quarantined"].iloc[0] == 1
