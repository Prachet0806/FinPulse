# simulation/impact_simulator.py

import pandas as pd
import logging
from decision_engine.strategy_library import STRATEGIES
from configs.settings import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def simulate_impact(df: pd.DataFrame):
    """
    Estimate financial impact of recommended strategies using vectorization.
    """

    logger.info("Running vectorized business impact simulation...")

    # Create a mapping of strategy names to their effectiveness and cost
    eff_map = {s: p["effectiveness"] for s, p in STRATEGIES.items()}
    cost_map = {s: p["cost"] for s, p in STRATEGIES.items()}

    # Map effectiveness and cost back to the dataframe
    effectiveness = df["recommended_strategy"].map(eff_map)
    costs = df["recommended_strategy"].map(cost_map)

    # Calculate retention gain for each row
    df["retention_gain"] = (
        df["churn_probability"] *
        df["estimated_clv"] *
        effectiveness
    )

    total_gain = df["retention_gain"].sum()
    total_cost = costs.sum()
    net_benefit = total_gain - total_cost

    # Clean up temp column
    df.drop(columns=["retention_gain"], inplace=True)

    return {
        "total_retention_gain": total_gain,
        "total_strategy_cost": total_cost,
        "net_benefit": net_benefit
    }