# simulation/strategy_breakdown.py

import pandas as pd


def strategy_impact_breakdown(df):
    """
    Per-strategy breakdown: counts, percents, and cost per strategy.
    Returns a DataFrame (not a bare Series).
    """
    strat = df.get("recommended_strategy") if isinstance(df, pd.DataFrame) else None
    if strat is None:
        return pd.DataFrame(columns=["strategy", "count", "percent", "cost_per_strategy"])
    try:
        from decision_engine.strategy_library import STRATEGIES

        cost_map = {s: p["cost"] for s, p in STRATEGIES.items()}
    except ImportError:
        cost_map = {}

    counts = strat.value_counts()
    total = counts.sum()
    rows = [
        {
            "strategy": name,
            "count": int(cnt),
            "percent": round(float(cnt) / float(total), 4) if total else 0.0,
            "cost_per_strategy": cost_map.get(name),
        }
        for name, cnt in counts.items()
    ]
    return pd.DataFrame(rows, columns=["strategy", "count", "percent", "cost_per_strategy"])
