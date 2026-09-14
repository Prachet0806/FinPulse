import pytest
import pandas as pd
import numpy as np
from processing.feature_engineering import build_features, fintech_rename_columns, create_behavioral_features

def test_fintech_rename_columns():
    df = pd.DataFrame({
        "tenure": [1],
        "MonthlyCharges": [100],
        "TotalCharges": [1000]
    })
    df_renamed = fintech_rename_columns(df)
    assert "account_tenure_months" in df_renamed.columns
    assert "monthly_spend" in df_renamed.columns
    assert "lifetime_value" in df_renamed.columns

def test_division_by_zero_fix():
    """Verify that tenure=0 is fail-closed: NaN + flag, never inf."""
    df = pd.DataFrame({
        "account_tenure_months": [0, 10],
        "lifetime_value": [100, 100],
        "monthly_spend": [10, 10]
    })

    df_features = create_behavioral_features(df)

    # Check the tenure=0 case: NaN + flagged, not silent 0/inf
    assert pd.isna(df_features.iloc[0]["avg_revenue_per_month"])
    assert df_features.iloc[0]["is_zero_tenure"] == 1
    assert np.isfinite(df_features["avg_revenue_per_month"].dropna()).all()
    # Check regular case
    assert df_features.iloc[1]["avg_revenue_per_month"] == 10

def test_tenure_segmentation():
    """Verify that tenure categories are assigned correctly."""
    df = pd.DataFrame({
        "account_tenure_months": [5, 15, 30, 60],
        "lifetime_value": [100, 200, 300, 400],
        "monthly_spend": [10, 10, 10, 10]
    })
    
    df_features = create_behavioral_features(df)
    assert list(df_features["tenure_segment"]) == ["new", "early", "mid", "loyal"]
