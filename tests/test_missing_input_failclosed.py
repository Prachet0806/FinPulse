import numpy as np
import pandas as pd

from decision_engine.optimizer import optimize_strategies_vectorized
from decision_engine.risk_adjustment import calculate_risk_penalty
from decision_engine.strategy_library import STRATEGIES
from intelligence.clv.clv_model import estimate_clv
from intelligence.customer_priority import compute_priority
from simulation.impact_simulator import simulate_impact


def test_missing_clv_inputs_produce_zero_and_quarantine_flag():
    out = estimate_clv(pd.DataFrame({"customerID": ["a", "b"]}))
    assert out["estimated_clv"].tolist() == [0.0, 0.0]
    assert out["is_clv_quarantined"].tolist() == [1, 1]


def test_invalid_arpu_and_churn_cannot_create_clv_or_priority():
    df = pd.DataFrame({
        "churn_probability": [0.5, np.inf, 0.7],
        "avg_revenue_per_month": [-10.0, 100.0, 100.0],
    })
    out = compute_priority(estimate_clv(df))
    assert out["estimated_clv"].iloc[:2].tolist() == [0.0, 0.0]
    assert out["is_clv_quarantined"].iloc[:2].tolist() == [1, 1]
    assert out["priority_score"].iloc[:2].tolist() == [0.0, 0.0]
    assert out["priority_score"].iloc[2] > 0


def test_missing_priority_inputs_are_zero():
    out = compute_priority(pd.DataFrame({"customerID": ["a"]}))
    assert out["priority_score"].tolist() == [0.0]


def test_missing_risk_inputs_use_max_churn_penalty_and_zero_clv_scaling():
    rows = pd.DataFrame({"customerID": ["a"]})
    params = STRATEGIES["loan_offer"]
    flat = calculate_risk_penalty(rows, "loan_offer", params)
    scaled = calculate_risk_penalty(rows, "loan_offer", params, use_clv_scaling=True)
    assert flat.iloc[0] == scaled.iloc[0]
    assert np.isfinite(flat.iloc[0])


def test_missing_money_inputs_never_generate_strategy_or_simulated_gain():
    rows = pd.DataFrame({"customerID": ["a", "b"]})
    recommended = optimize_strategies_vectorized(rows)
    assert recommended["recommended_strategy"].tolist() == ["no_action", "no_action"]

    impact = simulate_impact(pd.DataFrame({"recommended_strategy": ["loan_offer", "no_action"]}))
    assert impact["n_quarantined"] == 2
    assert impact["total_retention_gain"] == 0.0
    assert impact["total_strategy_cost"] == 0.0
