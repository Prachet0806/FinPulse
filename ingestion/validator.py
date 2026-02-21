# ingestion/validator.py

import pandas as pd


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data validation and cleaning.

    Steps:
    - Remove duplicates
    - Handle missing values
    - Ensure correct data types

    Returns:
        Cleaned dataframe
    """

    print("[INFO] Starting data validation...")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(method="ffill")

    # Convert TotalCharges to numeric if needed
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    print("[INFO] Data validation complete.")
    return df