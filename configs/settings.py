# configs/settings.py

import logging
import logging.config
import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths (single source of truth)
RAW_DATA_PATH = Path(
    os.getenv("FINPULSE_RAW_PATH", BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv")
).resolve()
MODELS_DIR = (BASE_DIR / "models").resolve()
PROCESSED_DIR = (BASE_DIR / "data" / "processed").resolve()

# Model Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
DISCOUNT_RATE = 0.1

# Decision Engine Constraints
MONTHLY_BUDGET = 15000  # Global budget for retention campaigns

# Segmentation Parameters
DEFAULT_CLUSTERS = 4

# --- Centralized magic numbers (permissive fail-closed plan) ---
FINTECH_SCALE = 10
ENGAGEMENT_WTS = (0.5, 0.5)
TENURE_BINS = [0, 12, 24, 48, 72]
CLV_MIN_DENOM = 0.05
EPS = 1e-9
RISK_SLOPES = {"credit_limit_increase": 200, "loan_offer": 100}

# Ingestion schema
REQUIRED_COLUMNS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]

# Logging Config
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv("FINPULSE_LOG_LEVEL", "INFO")


def ensure_dirs() -> None:
    """Create runtime directories (models, processed) if missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Call only from entrypoints (scripts/*, dashboard)."""
    lvl = (level or LOG_LEVEL).upper()
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"finpulse": {"format": LOG_FORMAT}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "finpulse",
                    "level": lvl,
                }
            },
            "root": {"handlers": ["console"], "level": lvl},
        }
    )
