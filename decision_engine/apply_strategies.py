# decision_engine/apply_strategies.py

from decision_engine.optimizer import optimize_strategies_vectorized


def apply_strategies(df):
    """
    Assign recommended action to each customer using vectorized engine.
    """

    df = optimize_strategies_vectorized(df)
    return df