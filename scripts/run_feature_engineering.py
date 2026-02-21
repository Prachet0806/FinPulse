# scripts/run_feature_engineering.py

from ingestion.loader import load_data
from ingestion.validator import validate_data
from processing.feature_engineering import build_features


def main():
    raw_path = "data/raw/telco_churn.csv"

    df = load_data(raw_path)
    df = validate_data(df)

    df = build_features(df)

    print(df.head())


if __name__ == "__main__":
    main()