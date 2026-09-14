# scripts/run_predictive_intelligence.py

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.settings import RAW_DATA_PATH, setup_logging, ensure_dirs

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FinPulse predictive intelligence")
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
    from intelligence.segmentation.segmenter import segment_customers
    from intelligence.segmentation.segment_labels import label_segments

    from intelligence.churn.churn_model import train_churn_model
    from intelligence.churn.churn_scoring import score_churn
    from intelligence.clv.clv_model import estimate_clv
    from intelligence.customer_priority import compute_priority

    try:
        df = load_data(args.data_path)
        df, _, _ = validate_data(df, return_report=True)
        df = build_features(df)

        df = segment_customers(df)
        df = label_segments(df)

        model = train_churn_model(df)
        df = score_churn(model, df)

        df = estimate_clv(df)
        df = compute_priority(df)
    except Exception:
        logger.exception("Fatal predictive-intelligence error")
        sys.exit(1)

    print(df[[
        "segment_label",
        "churn_probability",
        "estimated_clv",
        "priority_score"
    ]].head())


if __name__ == "__main__":
    main()
