import pytest
import pandas as pd
import os
from ingestion.loader import load_data
from ingestion.validator import validate_data
from configs.settings import RAW_DATA_PATH

def test_load_data():
    """Verify that data can be loaded from the configured path."""
    df = load_data(RAW_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "tenure" in df.columns or "account_tenure_months" in df.columns

def test_validate_data():
    """Verify that data validation handles basic requirements."""
    df = pd.DataFrame({
        "customerID": ["1", "2"],
        "tenure": [1, 2],
        "MonthlyCharges": [10.0, 20.0],
        "TotalCharges": ["10.0", "40.0"],
        "Churn": ["No", "Yes"]
    })
    
    # Test for specific cleaning logic if it exists (e.g. TotalCharges being numeric)
    # The current validator just returns the df or prints info. 
    # Let's ensure it doesn't crash.
    df_valid = validate_data(df)
    assert len(df_valid) == 2
