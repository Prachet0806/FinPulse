# decision_engine/risk_adjustment.py

import pandas as pd

def calculate_risk_penalty(df: pd.DataFrame, strategy: str, params: dict) -> pd.Series:
    """
    Calculate the risk penalty for a given strategy per customer.
    Penalties dynamically scale with the baseline risk probability.
    """
    base_penalty = params.get("risk_penalty", 0.0)
    
    # Example: Strategy increases financial exposure if they churn anyway
    if strategy == "credit_limit_increase":
        # The penalty scales severely with churn probability
        return base_penalty + (df["churn_probability"] * 200)
    
    elif strategy == "loan_offer":
        return base_penalty + (df["churn_probability"] * 100)
    
    # Default is the static base penalty
    return pd.Series(base_penalty, index=df.index)
