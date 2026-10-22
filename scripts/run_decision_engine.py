"""Train on labeled data and display held-out strategy scenarios."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.settings import RAW_DATA_PATH, setup_logging
from pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(RAW_DATA_PATH))
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()
    setup_logging(args.log_level)
    try:
        df, _, _, _ = PipelineOrchestrator(args.data_path).run_full_pipeline(mode="train")
    except Exception:
        logger.exception("Decision evaluation failed")
        sys.exit(1)
    print(df[["segment_label", "churn_probability", "estimated_clv", "recommended_strategy"]].head())


if __name__ == "__main__":
    main()
