"""Score an unlabeled customer CSV using the approved local model bundle."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.settings import setup_logging
from pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()
    setup_logging(args.log_level)
    try:
        df, impact, roi, _ = PipelineOrchestrator(args.data_path).run_full_pipeline(
            checkpoint_dir=args.checkpoint_dir, mode="inference"
        )
    except Exception:
        logger.exception("Inference failed")
        sys.exit(1)
    print(df[["customerID", "churn_probability", "estimated_clv", "recommended_strategy"]].head())
    print("Unvalidated scenario estimate:", impact)
    print("Gross scenario ROI:", roi)


if __name__ == "__main__":
    main()
