# configs/settings.py

import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
DISCOUNT_RATE = 0.1

# Decision Engine Constraints
MONTHLY_BUDGET = 15000  # Global budget for retention campaigns

# Segmentation Parameters
DEFAULT_CLUSTERS = 4

# Logging Config
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
