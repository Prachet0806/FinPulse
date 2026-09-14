# simulation/impact_simulator.py

import logging

import pandas as pd

from configs.settings import EPS
from decision_engine.risk_adjustment import calculate_risk_penalty
from decision_engine.strategy_library import STRATEGIES

logger = logging.getLogger(__name__)


def simulate_impact(df: pd.DataFrame):
    """
    Estimate financial impact of recommended strategies (no in-place mutation).

    Reports both gross gain and risk-adjusted net so the dashboard can
    switch to the risk-adjusted view. Unknown strategies / NaN churn / NaN
    CLV are quarantined (warned) and excluded from totals instead of being
    silently undercounted via skipna.
    """
    work = df.copy()
    logger.info("Running vectorized business impact simulation...")

    # Validate strategies
    valid_strategies = set(STRATEGIES.keys())
    strat = work.get("recommended_strategy")
    if strat is None:
        logger.warning("Simulator quarantine: recommended_strategy missing; all zero")
        return {
            "total_retention_gain": 0.0,
            "total_strategy_cost": 0.0,
            "net_benefit": 0.0,
            "risk_penalty_total": 0.0,
            "net_risk_adjusted": 0.0,
            "n_quarantined": int(len(work)),
        }
    unknown = ~strat.isin(valid_strategies)
    churn = pd.to_numeric(work.get("churn_probability"), errors="coerce")
    clv = pd.to_numeric(work.get("estimated_clv"), errors="coerce")
    bad = unknown | churn.isna() | clv.isna()
    n_quarantined = int(bad.sum())
    if n_quarantined > 0:
        logger.warning(
            "Simulator quarantine: %d rows with unknown strategy or NaN churn/CLV excluded",
            n_quarantined,
        )
    good = work[~bad].copy()
    if good.empty:
        return {
            "total_retention_gain": 0.0,
            "total_strategy_cost": 0.0,
            "net_benefit": 0.0,
            "risk_penalty_total": 0.0,
            "net_risk_adjusted": 0.0,
            "n_quarantined": n_quarantined,
        }

    # Create a mapping of strategy names to their effectiveness and cost
    eff_map = {s: p["effectiveness"] for s, p in STRATEGIES.items()}
    cost_map = {s: p["cost"] for s, p in STRATEGIES.items()}

    effectiveness = good["recommended_strategy"].map(eff_map)
    costs = good["recommended_strategy"].map(cost_map)
    churn_g = pd.to_numeric(good["churn_probability"], errors="coerce").fillna(0.0)
    clv_g = pd.to_numeric(good["estimated_clv"], errors="coerce").fillna(0.0)

    # Gross retention gain for each row
    retention_gain = churn_g * clv_g * effectiveness

    total_gain = float(retention_gain.sum())
    total_cost = float(costs.sum())
    net_benefit = total_gain - total_cost

    # Risk-adjusted view (parity with optimizer)
    risk_total = 0.0
    for strategy, params in STRATEGIES.items():
        sub = good[good["recommended_strategy"] == strategy]
        if sub.empty or strategy == "no_action":
            continue
        pen = calculate_risk_penalty(sub, strategy, params)
        risk_total += float(pd.to_numeric(pen, errors="coerce").fillna(0.0).sum())
    net_risk_adjusted = total_gain - total_cost - risk_total
    if abs(risk_total) > EPS:
        logger.info(
            "Simulator: gross=%.2f cost=%.2f risk=%.2f net_risk_adjusted=%.2f",
            total_gain,
            total_cost,
            risk_total,
            net_risk_adjusted,
        )

    return {
        "total_retention_gain": round(total_gain, 2),
        "total_strategy_cost": round(total_cost, 2),
        "net_benefit": round(net_benefit, 2),
        "risk_penalty_total": round(risk_total, 2),
        "net_risk_adjusted": round(net_risk_adjusted, 2),
        "n_quarantined": n_quarantined,
    }
