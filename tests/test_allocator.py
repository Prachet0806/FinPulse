import math

import pandas as pd

from configs import settings
from decision_engine.portfolio_allocator import allocate_portfolio


def _inputs(n=8, net=500.0, cost=100.0):
    df = pd.DataFrame({"x": range(n)})
    idx = df.index
    return (
        df,
        {"loan_offer": pd.Series([net] * n, index=idx)},
        {"loan_offer": pd.Series([cost] * n, index=idx)},
    )


def test_allocator_returns_summary_and_rounds():
    df, nets, costs = _inputs()
    out, summary = allocate_portfolio(df, nets, costs)
    assert "recommended_strategy" in out.columns
    assert summary["total_cost"] <= settings.MONTHLY_BUDGET
    assert summary["total_cost"] == round(summary["total_cost"], 2)


def test_allocator_zero_budget_blocks_even_zero_cost():
    df = pd.DataFrame({"x": [1, 2]})
    nets = {"credit_limit_increase": pd.Series([500.0, 500.0], index=df.index)}
    costs = {"credit_limit_increase": pd.Series([0.0, 0.0], index=df.index)}
    old = settings.MONTHLY_BUDGET
    settings.MONTHLY_BUDGET = 0
    try:
        out, summary = allocate_portfolio(df, nets, costs)
    finally:
        settings.MONTHLY_BUDGET = old
    assert (out["recommended_strategy"] == "no_action").all()


def test_allocator_small_n_cap_and_dup_index():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[7, 7, 7])
    nets = {"loan_offer": pd.Series([500.0, 400.0, 300.0], index=df.index)}
    costs = {"loan_offer": pd.Series([50.0, 50.0, 50.0], index=df.index)}
    out, summary = allocate_portfolio(df, nets, costs)
    # max(1, ceil(3*0.10)) = 1 slot
    assert (out["recommended_strategy"] == "loan_offer").sum() <= 1
    assert not bool(out.index.duplicated().any())


def test_allocator_inf_tiebreak_prefers_higher_net():
    df = pd.DataFrame({"x": [1, 2]})
    # Both zero-cost -> inf ROI; cap ceil(2*0.15)=1 so only the higher net wins
    nets = {"credit_limit_increase": pd.Series([900.0, 100.0], index=df.index)}
    costs = {"credit_limit_increase": pd.Series([0.0, 0.0], index=df.index)}
    old = settings.MONTHLY_BUDGET
    settings.MONTHLY_BUDGET = math.inf  # budget irrelevant for zero-cost
    try:
        out, summary = allocate_portfolio(df, nets, costs)
    finally:
        settings.MONTHLY_BUDGET = old
    assert summary["total_net_value"] == 900.0
    assert out["recommended_strategy"].iloc[0] == "credit_limit_increase"
