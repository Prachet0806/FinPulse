# decision_engine/optimizer.py

import pandas as pd
import logging
from decision_engine.strategy_library import STRATEGIES
from decision_engine.eligibility import apply_eligibility_mask
from decision_engine.risk_adjustment import calculate_risk_penalty
from decision_engine.portfolio_allocator import allocate_portfolio

logger = logging.getLogger(__name__)

def optimize_strategies_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select best strategies across the portfolio using a V2 enterprise framework:
    1. Base Expected Value calculation (Gain - Cost)
    2. Risk Adjustment (Probability-scaled penalties)
    3. Eligibility Rules (Business logic exclusions)
    4. Portfolio Allocation (Budget & Strategy Caps)
    """
    logger.info("Running V2 Multi-Layer Decision Optimization...")
    
    strategy_net_values = {}
    strategy_costs = {}

    for strategy, params in STRATEGIES.items():
        if strategy == "no_action":
            # Baseline value is 0
            strategy_net_values[strategy] = pd.Series(0.0, index=df.index)
            strategy_costs[strategy] = pd.Series(0.0, index=df.index)
            continue
            
        cost = params["cost"]
        effectiveness = params["effectiveness"]
        
        # 1. Base Expected Value = Gain - Cost
        base_net_value = (df["churn_probability"] * df["estimated_clv"] * effectiveness) - cost
        
        # 2. Risk Adjustment
        risk_penalty = calculate_risk_penalty(df, strategy, params)
        adjusted_value = base_net_value - risk_penalty
        
        # 3. Eligibility Rules (masks invalid targets with -inf)
        final_value = apply_eligibility_mask(df, strategy, adjusted_value)
        
        strategy_net_values[strategy] = final_value
        strategy_costs[strategy] = pd.Series(cost, index=df.index)

    # 4. Portfolio Allocation
    df = allocate_portfolio(df, strategy_net_values, strategy_costs)
    
    return df