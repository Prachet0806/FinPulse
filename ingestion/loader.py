# ingestion/loader.py

import logging
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file (permissive fail-closed: raise only on fatal).

    Args:
        path (str): Path to raw dataset

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if path is None or (isinstance(path, str) and not path.strip()):
        raise ValueError("Dataset path must be a non-empty string")
    file_path = Path(path).resolve()
    # Basic traversal/validity guard: must be an existing file
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            na_values=[" ", "", "NA"],
            dtype={"customerID": str, "Churn": str},
            keep_default_na=True,
        )
    except (ParserError, EmptyDataError, UnicodeDecodeError) as exc:
        logger.exception("Failed to parse dataset at %s", file_path)
        raise ValueError(f"Unable to load dataset at {path}: {exc}") from exc

    logger.info(
        "Loaded dataset with shape: %s | nulls: %d | dups: %d",
        df.shape,
        int(df.isna().sum().sum()),
        int(df.duplicated().sum()),
    )
    return df
