# scripts/run_ingestion.py

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for local execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.settings import RAW_DATA_PATH, setup_logging, ensure_dirs

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FinPulse ingestion")
    p.add_argument("--data-path", default=str(RAW_DATA_PATH))
    p.add_argument("--log-level", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    ensure_dirs()
    from ingestion.loader import load_data
    from ingestion.validator import validate_data

    try:
        df = load_data(args.data_path)
        clean_df, quarantine_df, report = validate_data(df, return_report=True)
    except Exception:
        logger.exception("Fatal ingestion error")
        sys.exit(1)
    logger.info("Ingestion report: %s", report)
    if not quarantine_df.empty:
        logger.warning("Quarantined %d rows", len(quarantine_df))
    print(clean_df.head())


if __name__ == "__main__":
    main()
