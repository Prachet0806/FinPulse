# ingestion/loader.py

import pandas as pd
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.

    Args:
        path (str): Path to raw dataset

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(file_path)

    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    return df