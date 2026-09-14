import pandas as pd
import pytest


@pytest.fixture
def raw_telco_frame():
    return pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "tenure": [5, 20, 40],
        "MonthlyCharges": [50.0, 80.0, 30.0],
        "TotalCharges": ["250.0", "1600.0", "1200.0"],
        "Churn": ["No", "Yes", "No"],
    })


@pytest.fixture
def featured_frame():
    # Post-feature-engineering shape for decision/sim tests
    return pd.DataFrame({
        "customerID": [f"C{i}" for i in range(6)],
        "account_tenure_months": [24, 24, 24, 2, 36, 36],
        "monthly_spend": [500.0] * 6,
        "lifetime_value": [5000.0] * 6,
        "avg_revenue_per_month": [200.0] * 6,
        "engagement_score": [0.5] * 6,
        "churn_probability": [0.9, 0.8, 0.1, 0.9, 0.9, float("nan")],
        "estimated_clv": [5000.0, 5000.0, 5000.0, 5000.0, 100.0, 5000.0],
        "tenure_segment_new": [0, 0, 0, 1, 0, 0],
    })
