# intelligence/churn/churn_model.py

import pandas as pd
import logging
import os
import joblib
from configs.settings import RANDOM_SEED, TEST_SIZE, LOG_FORMAT, LOG_LEVEL, MODELS_DIR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def train_churn_model(df: pd.DataFrame, save_model=True):
    """
    Train churn prediction model and optionally save it.
    """

    logger.info("Training churn model...")

    # Target variable
    if "Churn" not in df.columns:
        # Fallback if already processed/renamed (though Churn is usually kept)
        logger.warning("'Churn' column not found, checking for case variation")
        target_cols = [c for c in df.columns if c.lower() == "churn"]
        if not target_cols:
             raise ValueError("Churn column not found in dataset")
        target_col = target_cols[0]
    else:
        target_col = "Churn"

    y = df[target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})

    # Use numeric and dummy-encoded features
    X = df.select_dtypes(include=["int64", "float64", "uint8", "int32"])
    
    # Drop target and irrelevant IDs
    cols_to_drop = [target_col, "customerID", "segment_id"]
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    model = RandomForestClassifier(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    logger.info(f"Churn model training complete. Accuracy proxy (test): {accuracy:.2f}")

    if save_model:
        model_path = os.path.join(MODELS_DIR, "churn_model.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

    return model


def load_churn_model():
    """Load the churn model from disk."""
    model_path = os.path.join(MODELS_DIR, "churn_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None