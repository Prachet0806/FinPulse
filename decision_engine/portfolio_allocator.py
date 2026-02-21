# decision_engine/portfolio_allocator.py

import pandas as pd
import numpy as np
from decision_engine.strategy_library import STRATEGIES
from configs.settings import MONTHLY_BUDGET
import logging

logger = logging.getLogger(__name__)

def allocate_portfolio(df: pd.DataFrame, strategy_net_values: dict, strategy_costs: dict) -> pd.DataFrame:
    """
    Allocates strategies based on ROI (Net Value / Cost) subject to global budget
    and individual strategy capacity constraints.
    """
    logger.info("Executing Heuristic Portfolio Allocation (Knapsack Approximation)...")
    
    budget_remaining = MONTHLY_BUDGET
    df["recommended_strategy"] = "no_action"
    
    total_customers = len(df)
    assigned_counts = {s: 0 for s in STRATEGIES.keys()}
    
    records = []
    
    # 1. Flatten into (Customer, Strategy, NetValue, Cost, ROI) matrix
    for strategy in STRATEGIES.keys():
        if strategy == "no_action": continue
            
        net_val_series = strategy_net_values[strategy]
        cost_series = strategy_costs[strategy]
        
        # Only allocate if the transaction is ROI positive (>0 net value after penalties)
        valid_idx = net_val_series[net_val_series > 0].index
        
        for idx in valid_idx:
            net_val = net_val_series.loc[idx]
            cost = cost_series.loc[idx]
            roi = net_val / cost if cost > 0 else np.inf
            
            records.append({
                "idx": idx,
                "strategy": strategy,
                "net_value": net_val,
                "cost": cost,
                "roi": roi
            })
            
    if not records:
        logger.warning("No mathematically positive ROI strategies found for any customer.")
        return df
        
    # 2. Sort decisions by density (ROI)
    candidates = pd.DataFrame(records).sort_values(by="roi", ascending=False)
    
    assigned_customers = set()
    
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
            
        # Hard Stop: Capacity Limits (Caps)
        cap_pct = STRATEGIES[strat].get("cap_percent", 1.0)
        if assigned_counts[strat] >= int(total_customers * cap_pct):
            continue
            
        # Execute Strategy Assignment
        df.at[idx, "recommended_strategy"] = strat
        budget_remaining -= cost
        assigned_counts[strat] += 1
        assigned_customers.add(idx)
        
    logger.info(f"Allocation Complete. Remaining Budget: ${budget_remaining:,.2f}")
    return df
