# scripts/run_feature_engineering.py

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.settings import RAW_DATA_PATH, setup_logging, ensure_dirs

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FinPulse feature engineering")
    p.add_argument("--data-path", default=str(RAW_DATA_PATH))
    p.add_argument("--log-level", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    ensure_dirs()
    from ingestion.loader import load_data
    from ingestion.validator import validate_data
    from processing.feature_engineering import build_features

    try:
        df = load_data(args.data_path)
        df, _, _ = validate_data(df, return_report=True)
        df = build_features(df)
    except Exception:
        logger.exception("Fatal feature-engineering error")
        sys.exit(1)

    print(df.head())


if __name__ == "__main__":
    main()
