# intelligence/clv/clv_model.py

import logging
from configs.settings import DISCOUNT_RATE, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def estimate_clv(df):
    """
    Estimate CLV using the financial retention formula:
    CLV = (ARPU * Retention) / (1 + Discount - Retention)
    """

    logger.info("Estimating advanced CLV...")

    # Calculate retention from churn probability
    # Handle cases where churn_probability might be missing (fallback to 0)
    churn_prob = df.get("churn_probability", 0)
    retention = 1 - churn_prob

    # ARPU from behavioral features
    arpu = df["avg_revenue_per_month"]

    df["estimated_clv"] = (
        arpu * retention
    ) / (1 + DISCOUNT_RATE - retention)

    return df