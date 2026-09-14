# decision_engine/portfolio_allocator.py

import logging
import math

import numpy as np
import pandas as pd

from configs import settings as _settings
from configs.settings import EPS
from decision_engine.strategy_library import STRATEGIES

logger = logging.getLogger(__name__)


def allocate_portfolio(df: pd.DataFrame, strategy_net_values: dict, strategy_costs: dict):
    """
    Greedy ROI-density allocation subject to global budget and caps.

    Fail-closed: MONTHLY_BUDGET <= 0 blocks even zero-cost strategies
    (all no_action). Returns (df, allocation_summary).

    NOTE: greedy is a knapsack approximation (fast, minimal diff).
    An optimal LP solver (pulp) is tracked as V2 roadmap.
    """
    df = df.copy()
    # Guard against duplicate index (df.at would be ambiguous)
    if bool(df.index.duplicated().any()):
        logger.warning("Duplicate index detected; resetting index for allocation")
        df = df.reset_index(drop=True)
        strategy_net_values = {
            k: pd.Series(v.values, index=df.index) for k, v in strategy_net_values.items()
        }
        strategy_costs = {
            k: pd.Series(v.values, index=df.index) for k, v in strategy_costs.items()
        }

    budget_remaining = _settings.MONTHLY_BUDGET
    if budget_remaining <= 0:
        logger.warning("Zero/non-positive budget; all customers -> no_action")
        df["recommended_strategy"] = "no_action"
        summary = {
            "total_net_value": 0.0,
            "total_cost": 0.0,
            "budget_remaining": round(float(budget_remaining), 2),
            "assigned_counts": {"no_action": int(len(df))},
        }
        return df, summary

    df["recommended_strategy"] = "no_action"

    total_customers = len(df)
    assigned_counts = {s: 0 for s in STRATEGIES.keys()}

    records = []

    # 1. Flatten into (Customer, Strategy, NetValue, Cost, ROI) matrix
    for strategy in STRATEGIES.keys():
        if strategy == "no_action":
            continue
        if strategy not in strategy_net_values or strategy not in strategy_costs:
            logger.debug("Allocator skipping %s (no net/cost series provided)", strategy)
            continue

        net_val_series = strategy_net_values[strategy]
        cost_series = strategy_costs[strategy]

        # Only allocate if ROI positive (> EPS net value after penalties)
        valid_idx = net_val_series[net_val_series > EPS].index

        for idx in valid_idx:
            net_val = net_val_series.loc[idx]
            cost = cost_series.loc[idx]
            roi = net_val / cost if cost > 0 else np.inf

            records.append(
                {
                    "idx": idx,
                    "strategy": strategy,
                    "net_value": round(float(net_val), 2),
                    "cost": round(float(cost), 2),
                    "roi": roi,
                }
            )

    if not records:
        logger.warning("No mathematically positive ROI strategies found for any customer.")
        df["recommended_strategy"] = "no_action"
        summary = {
            "total_net_value": 0.0,
            "total_cost": 0.0,
            "budget_remaining": round(float(budget_remaining), 2),
            "assigned_counts": {"no_action": int(len(df))},
        }
        return df, summary

    # 2. Sort decisions by density (ROI), tie-break on net value
    candidates = pd.DataFrame(records).sort_values(
        by=["roi", "net_value"], ascending=[False, False]
    )

    assigned_customers = set()
    total_net = 0.0
    total_cost = 0.0

    # 3. Greedy Allocation respecting Constraints
    for _, row in candidates.iterrows():
        idx = row["idx"]
        strat = row["strategy"]
        cost = row["cost"]

        # Already processed this customer via a higher-ROI opportunity
        if idx in assigned_customers:
            continue

        # Hard Stop: Budget constraint
        if budget_remaining < cost:
            continue

        # Hard Stop: Capacity Limits (Caps), at least 1 slot when N small
        cap_pct = STRATEGIES[strat].get("cap_percent", 1.0)
        cap = max(1, math.ceil(total_customers * cap_pct))
        if assigned_counts[strat] >= cap:
            continue

        # Execute Strategy Assignment
        df.at[idx, "recommended_strategy"] = strat
        budget_remaining -= cost
        total_net += row["net_value"]
        total_cost += cost
        assigned_counts[strat] += 1
        assigned_customers.add(idx)

    assigned_counts["no_action"] = int(total_customers - len(assigned_customers))
    assert total_cost <= _settings.MONTHLY_BUDGET + EPS, "Allocator overspent budget"
    summary = {
        "total_net_value": round(float(total_net), 2),
        "total_cost": round(float(total_cost), 2),
        "budget_remaining": round(float(budget_remaining), 2),
        "assigned_counts": {k: int(v) for k, v in assigned_counts.items()},
    }
    logger.info(
        "Allocation Complete. Net=$%.2f Cost=$%.2f Remaining Budget: $%.2f",
        total_net,
        total_cost,
        budget_remaining,
    )
    return df, summary
