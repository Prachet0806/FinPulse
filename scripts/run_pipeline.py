import logging
import sys
from pathlib import Path

# Add project root to path for local execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.settings import LOG_FORMAT, LOG_LEVEL
from pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def main():
    orchestrator = PipelineOrchestrator()
    df, _, _, _ = orchestrator.run_full_pipeline()

    print("\nSample Output:")
    print(df[[
        "segment_label",
        "churn_probability",
        "estimated_clv",
        "recommended_strategy"
    ]].head())


if __name__ == "__main__":
    main()