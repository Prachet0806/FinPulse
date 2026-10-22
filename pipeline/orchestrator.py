"""Explicit training and inference entry points for the FinPulse pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd

from configs.settings import PROCESSED_DIR, RAW_DATA_PATH, ensure_dirs
from ingestion.loader import load_data
from pipeline.model_bundle import score_new_customers, train_and_evaluate

logger = logging.getLogger(__name__)


def _write_quarantine(quarantine_df: pd.DataFrame, stage: str) -> None:
    if quarantine_df is None or quarantine_df.empty:
        return
    ensure_dirs()
    path = Path(PROCESSED_DIR) / (
        f"quarantine_{stage}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.csv"
    )
    quarantine_df.to_csv(path, index=False)
    logger.warning("Quarantined %d rows (%s) -> %s", len(quarantine_df), stage, path)


class PipelineOrchestrator:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.impact_metrics = None
        self.roi = None
        self.strategy_breakdown = None
        self.validation_report = None

    def run_full_pipeline(self, checkpoint_dir=None, *, mode="inference"):
        """Train/evaluate on a labeled holdout or score an unlabeled cohort."""
        if mode == "inference" and self.data_path is None:
            raise ValueError("Inference requires an explicit customer data path")
        raw = load_data(self.data_path or RAW_DATA_PATH)
        if mode == "train":
            bundle, result, quarantine = train_and_evaluate(raw)
            self.validation_report = bundle["metadata"]["validation_report"]
            _write_quarantine(quarantine, "training")
        elif mode == "inference":
            result, quarantine, self.validation_report = score_new_customers(raw)
            _write_quarantine(quarantine, "inference")
        else:
            raise ValueError("mode must be 'train' or 'inference'")
        self.df, self.impact_metrics, self.roi, self.strategy_breakdown = result
        if checkpoint_dir:
            out = Path(checkpoint_dir)
            out.mkdir(parents=True, exist_ok=True)
            self.df.to_parquet(out / "processed.parquet", index=False)
        return result
