# simulation/strategy_breakdown.py

def strategy_impact_breakdown(df):
    """
    Count customers per recommended strategy.
    """

    breakdown = df["recommended_strategy"].value_counts()

    return breakdown