# scripts/run_segmentation.py

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.settings import RAW_DATA_PATH, setup_logging, ensure_dirs

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FinPulse segmentation")
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
    from intelligence.segmentation.segment_insights import generate_segment_insights

    try:
        df = load_data(args.data_path)
        df, _, _ = validate_data(df, return_report=True)
        df = build_features(df)

        df = segment_customers(df)
        df = label_segments(df)

        insights = generate_segment_insights(df)
    except Exception:
        logger.exception("Fatal segmentation error")
        sys.exit(1)

    print(df[["segment_id", "segment_label"]].head())
    print("\nSegment Insights:")
    print(insights)


if __name__ == "__main__":
    main()
