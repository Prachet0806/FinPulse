import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for local execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.settings import setup_logging, ensure_dirs
from pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FinPulse full pipeline")
    p.add_argument("--data-path", default=None)
    p.add_argument("--log-level", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    ensure_dirs()
    kwargs = {}
    if args.data_path:
        kwargs["data_path"] = args.data_path
    try:
        orchestrator = PipelineOrchestrator(**kwargs)
        df, _, _, _ = orchestrator.run_full_pipeline()
    except Exception:
        logger.exception("Fatal pipeline error")
        sys.exit(1)

    print("\nSample Output:")
    print(df[[
        "segment_label",
        "churn_probability",
        "estimated_clv",
        "recommended_strategy"
    ]].head())


if __name__ == "__main__":
    main()
