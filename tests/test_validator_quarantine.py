import pandas as pd

from ingestion.validator import validate_data


def test_quarantine_bad_rows_continue():
    df = pd.DataFrame({
        "customerID": ["A", None, "C", "A"],
        "tenure": [10, 5, -3, 10],
        "MonthlyCharges": [50.0, 20.0, 30.0, 50.0],
        "TotalCharges": ["500.0", "100.0", "90.0", "500.0"],
        "Churn": ["No", "Yes", "Maybe", "No"],
    })
    clean, quarantine, report = validate_data(df, return_report=True)
    # None ID + (negative tenure / bad Churn) quarantined; dup A collapsed
    assert len(quarantine) == 2
    assert len(clean) == 1
    assert report["n_output"] == 1
    # Backward-compat wrapper still returns just the frame
    assert len(validate_data(df)) == 1


def test_totalcharges_coerce_then_median():
    df = pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "tenure": [10, 20, 30],
        "MonthlyCharges": [10.0, 20.0, 30.0],
        "TotalCharges": [" 100.0 ", "abc", "300.0"],
        "Churn": ["No", "No", "Yes"],
    })
    clean, quarantine, report = validate_data(df, return_report=True)
    assert len(quarantine) == 0  # "abc" is cleanable, not structural
    assert clean["TotalCharges"].isna().sum() == 0
    assert "is_totalcharges_missing" in clean.columns
    assert clean.loc[clean["customerID"] == "2", "is_totalcharges_missing"].iloc[0] == 1
