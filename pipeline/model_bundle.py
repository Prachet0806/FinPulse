"""Versioned training and inference workflow for FinPulse."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.settings import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, TEST_SIZE
from decision_engine.apply_strategies import apply_strategies
from ingestion.validator import validate_data
from intelligence.churn.churn_model import train_churn_model
from intelligence.churn.churn_scoring import score_churn
from intelligence.clv.clv_model import estimate_clv
from intelligence.customer_priority import compute_priority
from intelligence.segmentation.segment_labels import label_segments
from intelligence.segmentation.segmenter import apply_segment_models, segment_customers
from processing.feature_engineering import FeatureTransformer
from simulation.impact_simulator import simulate_impact
from simulation.roi_calculator import compute_roi
from simulation.strategy_breakdown import strategy_impact_breakdown

logger = logging.getLogger(__name__)
BUNDLE_VERSION = 1
BUNDLE_PATH = Path(MODELS_DIR) / "finpulse_bundle.joblib"
EVALUATION_PATH = Path(PROCESSED_DIR) / "evaluation.parquet"


def _score_features(features: pd.DataFrame, model) -> tuple:
    scored = score_churn(model, features)
    scored = estimate_clv(scored)
    scored = compute_priority(scored)
    scored = apply_strategies(scored)
    impact = simulate_impact(scored)
    return scored, impact, compute_roi(impact), strategy_impact_breakdown(scored)


def _publish_bundle(bundle: dict, path: Path | None = None) -> None:
    path = BUNDLE_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(bundle, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_bundle(path: Path | None = None) -> dict:
    path = BUNDLE_PATH if path is None else path
    if not path.is_file():
        raise FileNotFoundError(f"No trained model bundle at {path}; run training first")
    bundle = joblib.load(path)  # Only load artifacts from the trusted local model directory.
    if not isinstance(bundle, dict) or bundle.get("version") != BUNDLE_VERSION:
        raise ValueError("Unsupported FinPulse model bundle version")
    required = {"transformer", "segments", "churn_model", "feature_columns", "metadata"}
    if not required <= bundle.keys():
        raise ValueError("Incomplete FinPulse model bundle")
    model_cols = list(getattr(bundle["churn_model"], "feature_names_in_", []))
    if model_cols != bundle["feature_columns"]:
        raise ValueError("Model and feature schema do not match")
    return bundle


def train_and_evaluate(raw: pd.DataFrame, *, publish: bool = True) -> tuple:
    """Fit on one labeled cohort and score only its untouched holdout."""
    clean, quarantine, report = validate_data(
        raw, return_report=True, require_target=True, impute=False
    )
    if clean.empty:
        raise ValueError("No valid labeled rows remain after validation")
    counts = clean["Churn"].value_counts()
    if len(counts) != 2 or int(counts.min()) < 2:
        raise ValueError("Training requires at least two rows in each churn class")
    train_raw, holdout_raw = train_test_split(
        clean, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=clean["Churn"]
    )
    transformer = FeatureTransformer().fit(train_raw)
    train_features = transformer.transform(train_raw)
    holdout_features = transformer.transform(holdout_raw)
    train_features, segments = segment_customers(
        train_features, save_models=False, return_models=True
    )
    if segments[0] is None:
        raise ValueError("Too few training rows to fit segmentation")
    holdout_features = apply_segment_models(holdout_features, segments)
    train_features = label_segments(train_features)
    holdout_features = label_segments(holdout_features)
    model = train_churn_model(
        train_features, save_model=False, validation_df=holdout_features
    )
    scored, impact, roi, breakdown = _score_features(holdout_features, model)
    run_id = uuid4().hex
    scored["model_run_id"] = run_id
    metadata = {
        "run_id": run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "random_holdout_proxy",
        "n_train": int(len(train_raw)),
        "n_holdout": int(len(holdout_raw)),
        "validation_report": report,
        "metrics": model.finpulse_metrics_,
        "financial_status": "unvalidated_scenario",
    }
    bundle = {
        "version": BUNDLE_VERSION,
        "transformer": transformer,
        "segments": segments,
        "churn_model": model,
        "feature_columns": list(model.feature_names_in_),
        "metadata": metadata,
    }
    if publish:
        EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=EVALUATION_PATH.parent, suffix=".parquet")
        os.close(fd)
        try:
            scored.to_parquet(temp_path, index=False)
            os.replace(temp_path, EVALUATION_PATH)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        _publish_bundle(bundle)
    logger.info("Training complete: %s", metadata)
    return bundle, (scored, impact, roi, breakdown), quarantine


def score_new_customers(raw: pd.DataFrame, bundle: dict | None = None) -> tuple:
    """Score an unlabeled portfolio with a previously trained model bundle."""
    bundle = load_bundle() if bundle is None else bundle
    clean, quarantine, report = validate_data(
        raw, return_report=True, require_target=False, impute=False
    )
    if clean.empty:
        raise ValueError("No valid inference rows remain after validation")
    clean = clean.drop(columns=["Churn"], errors="ignore")
    features = bundle["transformer"].transform(clean)
    report["unseen_categories"] = {
        col: int(features[f"is_{col}_unseen"].sum())
        for col in bundle["transformer"].categories
    }
    features = apply_segment_models(features, bundle["segments"])
    features = label_segments(features)
    features["model_run_id"] = bundle["metadata"]["run_id"]
    logger.info("Inference validation report: %s", report)
    return _score_features(features, bundle["churn_model"]), quarantine, report
