# decision_engine/portfolio_allocator.py

import logging
import math

import numpy as np
import pandas as pd

from configs import settings as _settings
from configs.settings import EPS
from decision_engine.strategy_library import STRATEGIES

logger = logging.getLogger(__name__)


def _exact_selection(candidates: pd.DataFrame, total_customers: int, budget: float):
    """Solve the one-action-per-customer budgeted assignment for small cohorts."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n = len(candidates)
    customer_codes = pd.Categorical(candidates["idx"]).codes
    strategies = [s for s in STRATEGIES if s != "no_action"]
    strategy_codes = pd.Categorical(candidates["strategy"], categories=strategies).codes
    cols = np.arange(n)
    rows = np.concatenate([
        customer_codes,
        np.full(n, total_customers),
        total_customers + 1 + strategy_codes,
    ])
    matrix = coo_matrix(
        (np.concatenate([np.ones(n), candidates["cost"].to_numpy(), np.ones(n)]),
         (rows, np.tile(cols, 3))),
        shape=(total_customers + 1 + len(strategies), n),
    ).tocsr()
    caps = [max(1, math.ceil(total_customers * STRATEGIES[s].get("cap_percent", 1)))
            for s in strategies]
    upper = np.array([1.0] * total_customers + [float(budget)] + caps)
    result = milp(
        c=-candidates["net_value"].to_numpy(dtype=float),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, -np.inf, upper),
        options={"time_limit": 10},
    )
    if result.x is None or not result.success:
        logger.warning("Exact allocation unavailable (%s); using greedy fallback", result.message)
        return None
    return candidates.iloc[np.flatnonzero(result.x > 0.5)]


def allocate_portfolio(
    df: pd.DataFrame, strategy_net_values: dict, strategy_costs: dict,
    *, method: str = "auto",
):
    """
    Budgeted allocation with exact small-cohort and greedy large-cohort modes.

    Fail-closed: MONTHLY_BUDGET <= 0 blocks even zero-cost strategies
    (all no_action). Returns (df, allocation_summary).

    Uses an exact integer solver for cohorts up to 200 customers; larger
    cohorts use the documented greedy approximation.
    """
    df = df.copy()
    if method not in {"auto", "exact", "greedy"}:
        raise ValueError("method must be auto, exact, or greedy")
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
    if math.isnan(float(budget_remaining)):
        raise ValueError("MONTHLY_BUDGET must be a valid number")
    if budget_remaining <= 0:
        logger.warning("Zero/non-positive budget; all customers -> no_action")
        df["recommended_strategy"] = "no_action"
        summary = {
            "total_net_value": 0.0,
            "total_cost": 0.0,
            "budget_remaining": round(float(budget_remaining), 2),
            "method": "none",
            "assigned_counts": {"no_action": int(len(df))},
        }
        return df, summary

    df["recommended_strategy"] = "no_action"

    total_customers = len(df)
    assigned_counts = {s: 0 for s in STRATEGIES.keys()}

    candidate_frames = []

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
        net = pd.to_numeric(net_val_series, errors="coerce")
        costs = pd.to_numeric(cost_series, errors="coerce")
        if bool((costs.dropna() < 0).any()):
            raise ValueError(f"Negative cost configured for {strategy}")
        valid = (net.round(2) > 0) & (net > EPS) & np.isfinite(net) & costs.notna() & np.isfinite(costs)
        if not bool(valid.any()):
            continue
        frame = pd.DataFrame({
            "idx": net.index[valid],
            "strategy": strategy,
            "net_value": net[valid].round(2).to_numpy(),
            "cost": costs[valid].round(2).to_numpy(),
        })
        frame["roi"] = np.where(
            frame["cost"] > 0,
            frame["net_value"] / frame["cost"].replace(0, np.nan),
            np.inf,
        )
        candidate_frames.append(frame)

    if not candidate_frames:
        logger.warning("No mathematically positive ROI strategies found for any customer.")
        df["recommended_strategy"] = "no_action"
        summary = {
            "total_net_value": 0.0,
            "total_cost": 0.0,
            "budget_remaining": round(float(budget_remaining), 2),
            "method": "none",
            "assigned_counts": {"no_action": int(len(df))},
        }
        return df, summary

    # 2. Sort decisions by density (ROI), tie-break on net value
    candidates = pd.concat(candidate_frames, ignore_index=True).sort_values(
        by=["roi", "net_value"], ascending=[False, False]
    )

    assigned_customers = set()
    total_net = 0.0
    total_cost = 0.0

    selection_method = "greedy"
    chosen = candidates
    if method == "exact" or (method == "auto" and total_customers <= 200):
        exact = _exact_selection(candidates, total_customers, float(budget_remaining))
        if exact is not None:
            chosen = exact
            selection_method = "exact"
        elif method == "exact":
            raise RuntimeError("Exact allocation did not complete")

    # 3. Execute selected assignments while enforcing constraints again.
    for row in chosen.itertuples(index=False):
        idx = row.idx
        strat = row.strategy
        cost = row.cost

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
        total_net += row.net_value
        total_cost += cost
        assigned_counts[strat] += 1
        assigned_customers.add(idx)

    assigned_counts["no_action"] = int(total_customers - len(assigned_customers))
    if total_cost > _settings.MONTHLY_BUDGET + EPS:
        raise RuntimeError("Allocator overspent budget")
    summary = {
        "total_net_value": round(float(total_net), 2),
        "total_cost": round(float(total_cost), 2),
        "budget_remaining": round(float(budget_remaining), 2),
        "method": selection_method,
        "assigned_counts": {k: int(v) for k, v in assigned_counts.items()},
    }
    logger.info(
        "Allocation Complete. Net=$%.2f Cost=$%.2f Remaining Budget: $%.2f",
        total_net,
        total_cost,
        budget_remaining,
    )
    return df, summary
