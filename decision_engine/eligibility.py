# decision_engine/eligibility.py

import pandas as pd
import numpy as np

def apply_eligibility_mask(df: pd.DataFrame, strategy: str, net_value_series: pd.Series) -> pd.Series:
    """
    Apply business rules to qualify/disqualify customers for certain strategies.
    If a customer is ineligible, their expected net value for that strategy rests at -inf.
    """
    adjusted_series = net_value_series.copy()
    
    # 1. Credit Limit Increase Rule: Requires tenure > 12 months & low churn risk
    if strategy == "credit_limit_increase":
        ineligible_mask = (df["account_tenure_months"] <= 12) | (df["churn_probability"] > 0.6)
        adjusted_series[ineligible_mask] = -np.inf
        
    # 2. Loan Offer Rule: Only eligible if Estimated CLV is high enough to justify
    elif strategy == "loan_offer":
        ineligible_mask = df["estimated_clv"] < 500
        adjusted_series[ineligible_mask] = -np.inf
        
    # 3. Cashback Offer Rule: Exclude purely 'new' users to avoid gamification
    elif strategy == "cashback_offer" and "tenure_segment_new" in df.columns:
        # Assuming dummy variables are 1/0
        ineligible_mask = df["tenure_segment_new"] == 1
        adjusted_series[ineligible_mask] = -np.inf
        
    return adjusted_series
