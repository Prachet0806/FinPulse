# scripts/run_ingestion.py

from ingestion.loader import load_data
from ingestion.validator import validate_data


def main():
    raw_path = "data/raw/telco_churn.csv"

    df = load_data(raw_path)
    df = validate_data(df)

    print(df.head())


if __name__ == "__main__":
    main()