# scripts/run_impact_simulation.py

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.settings import RAW_DATA_PATH, setup_logging, ensure_dirs

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FinPulse impact simulation")
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

    from decision_engine.apply_strategies import apply_strategies

    from simulation.impact_simulator import simulate_impact
    from simulation.roi_calculator import compute_roi
    from simulation.strategy_breakdown import strategy_impact_breakdown

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

        df = apply_strategies(df)

        impact = simulate_impact(df)
        roi = compute_roi(impact)
        breakdown = strategy_impact_breakdown(df)
    except Exception:
        logger.exception("Fatal impact-simulation error")
        sys.exit(1)

    print("\n=== BUSINESS IMPACT ===")
    print(impact)

    print("\nROI:", roi)

    print("\nStrategy Breakdown:")
    print(breakdown)


if __name__ == "__main__":
    main()
