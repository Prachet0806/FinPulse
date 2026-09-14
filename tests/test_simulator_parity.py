import pandas as pd

from simulation.impact_simulator import simulate_impact
from simulation.roi_calculator import compute_roi
from simulation.strategy_breakdown import strategy_impact_breakdown


def test_simulator_reports_gross_and_risk_adjusted():
    df = pd.DataFrame({
        "churn_probability": [0.9, 0.5],
        "estimated_clv": [1000.0, 1000.0],
        "recommended_strategy": ["loan_offer", "fee_waiver"],
    })
    before = df.copy()
    impact = simulate_impact(df)
    # No in-place mutation
    assert "retention_gain" not in df.columns
    assert list(df.columns) == list(before.columns)
    assert impact["total_retention_gain"] > 0
    # Risk-adjusted <= gross net (penalties only subtract)
    assert impact["net_risk_adjusted"] <= impact["net_benefit"]
    assert impact["risk_penalty_total"] >= 0


def test_simulator_quarantines_unknown_strategy_and_nan():
    df = pd.DataFrame({
        "churn_probability": [0.9, float("nan"), 0.5],
        "estimated_clv": [1000.0, 1000.0, 1000.0],
        "recommended_strategy": ["loan_offer", "loan_offer", "mystery_strategy"],
    })
    impact = simulate_impact(df)
    assert impact["n_quarantined"] == 2
    # Only the first row contributes
    assert impact["total_retention_gain"] == round(0.9 * 1000.0 * 0.22, 2)


def test_roi_zero_cost_semantics():
    assert compute_roi({"total_strategy_cost": 0, "total_retention_gain": 100}) == float("inf")
    assert compute_roi({"total_strategy_cost": 0, "total_retention_gain": 0}) == 0.0
    assert compute_roi({"total_strategy_cost": float("nan"), "total_retention_gain": 5}) == 0.0
    assert compute_roi({"total_strategy_cost": -10, "total_retention_gain": 5}) == 0.0


def test_breakdown_returns_dataframe():
    df = pd.DataFrame({"recommended_strategy": ["loan_offer", "loan_offer", "no_action"]})
    out = strategy_impact_breakdown(df)
    assert list(out.columns) == ["strategy", "count", "percent", "cost_per_strategy"]
    assert out["count"].sum() == 3
    assert abs(out["percent"].sum() - 1.0) < 1e-9
