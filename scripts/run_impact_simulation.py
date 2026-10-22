"""Train on labeled data and display held-out scenario economics."""

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
        _, impact, roi, breakdown = PipelineOrchestrator(args.data_path).run_full_pipeline(mode="train")
    except Exception:
        logger.exception("Impact evaluation failed")
        sys.exit(1)
    print("\n=== UNVALIDATED SCENARIO ESTIMATE ===")
    print(impact)
    print("\nGross ROI:", roi)
    print("\nStrategy Breakdown:")
    print(breakdown)


if __name__ == "__main__":
    main()
