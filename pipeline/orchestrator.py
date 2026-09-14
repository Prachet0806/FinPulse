import logging
import time
from pathlib import Path

import pandas as pd
from configs.settings import PROCESSED_DIR, RAW_DATA_PATH, ensure_dirs

from ingestion.loader import load_data
from ingestion.validator import validate_data
from processing.feature_engineering import build_features

from intelligence.segmentation.segmenter import segment_customers
from intelligence.segmentation.segment_labels import label_segments
from intelligence.segmentation.segment_insights import generate_segment_insights

from intelligence.churn.churn_model import train_churn_model, load_churn_model
from intelligence.churn.churn_scoring import score_churn
from intelligence.clv.clv_model import estimate_clv
from intelligence.customer_priority import compute_priority

from decision_engine.apply_strategies import apply_strategies

from simulation.impact_simulator import simulate_impact
from simulation.roi_calculator import compute_roi
from simulation.strategy_breakdown import strategy_impact_breakdown

logger = logging.getLogger(__name__)


def _write_quarantine(quarantine_df: pd.DataFrame, stage: str) -> None:
    if quarantine_df is None or quarantine_df.empty:
        return
    try:
        ensure_dirs()
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = Path(PROCESSED_DIR) / f"quarantine_{stage}_{ts}.csv"
        quarantine_df.to_csv(path, index=False)
        logger.warning("Quarantined %d rows (%s) -> %s", len(quarantine_df), stage, path)
    except OSError:
        logger.exception("Failed to write quarantine CSV for stage %s", stage)


class PipelineOrchestrator:
    """
    Central orchestrator for FinPulse data and intelligence pipelines.
    Provides a unified API for CLI scripts and the Dashboard.
    Permissive fail-closed: stages log + quarantine + continue; abort only
    if the frame is empty after quarantine.
    """

    def __init__(self, data_path=RAW_DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.churn_model = None
        self.impact_metrics = None
        self.roi = None
        self.strategy_breakdown = None

    def _guard_nonempty(self, stage: str):
        if self.df is None or len(self.df) == 0:
            raise ValueError(f"Pipeline aborted: empty dataframe after {stage} quarantine")

    def load_and_process(self):
        """Step 1 & 2: Ingestion & Feature Engineering"""
        t0 = time.perf_counter()
        try:
            logger.info("Starting ingestion from %s...", self.data_path)
            raw = load_data(self.data_path)
            clean, quarantine, report = validate_data(raw, return_report=True)
            logger.info("Validation report: %s", report)
            _write_quarantine(quarantine, "ingestion")
            self.df = clean.copy()
            self._guard_nonempty("ingestion")

            logger.info("Starting feature engineering...")
            self.df = build_features(self.df.copy()).copy()
        except Exception:
            logger.exception("load_and_process failed")
            raise
        finally:
            logger.info(
                "load_and_process done in %.2fs rows=%s",
                time.perf_counter() - t0,
                len(self.df) if self.df is not None else 0,
            )
        return self.df

    def run_segmentation(self, use_stored=False):
        """Step 3: Customer Segmentation"""
        t0 = time.perf_counter()
        try:
            if self.df is None:
                self.load_and_process()
            self.df = self.df.copy()

            logger.info("Starting segmentation (use_stored=%s)...", use_stored)
            self.df = segment_customers(self.df, save_models=not use_stored, use_stored=use_stored).copy()
            self.df = label_segments(self.df).copy()
        except Exception:
            logger.exception("run_segmentation failed")
            raise
        finally:
            logger.info(
                "run_segmentation done in %.2fs rows=%s",
                time.perf_counter() - t0,
                len(self.df) if self.df is not None else 0,
            )
        return self.df

    def run_predictive_intelligence(self, use_stored=False):
        """Step 4: Churn Modeling, CLV, and Priority"""
        t0 = time.perf_counter()
        try:
            if self.df is None or "segment_id" not in self.df.columns:
                self.run_segmentation(use_stored=use_stored)
            self.df = self.df.copy()

            logger.info("Starting predictive intelligence...")

            if use_stored:
                self.churn_model = load_churn_model()
                if self.churn_model is None:
                    logger.warning("No stored model found. Training new model.")
                    self.churn_model = train_churn_model(self.df)
            else:
                self.churn_model = train_churn_model(self.df)

            self.df = score_churn(self.churn_model, self.df).copy()
            self.df = estimate_clv(self.df).copy()
            self.df = compute_priority(self.df).copy()
        except Exception:
            logger.exception("run_predictive_intelligence failed")
            raise
        finally:
            logger.info(
                "run_predictive_intelligence done in %.2fs rows=%s",
                time.perf_counter() - t0,
                len(self.df) if self.df is not None else 0,
            )
        return self.df

    def run_decision_engine(self):
        """Step 5: Strategy Optimization"""
        t0 = time.perf_counter()
        try:
            if self.df is None or "churn_probability" not in self.df.columns:
                self.run_predictive_intelligence()
            self.df = self.df.copy()

            logger.info("Applying optimization strategies...")
            self.df = apply_strategies(self.df).copy()
        except Exception:
            logger.exception("run_decision_engine failed")
            raise
        finally:
            logger.info(
                "run_decision_engine done in %.2fs rows=%s",
                time.perf_counter() - t0,
                len(self.df) if self.df is not None else 0,
            )
        return self.df

    def run_simulation(self):
        """Step 6: Impact & ROI Simulation"""
        t0 = time.perf_counter()
        try:
            if self.df is None or "recommended_strategy" not in self.df.columns:
                self.run_decision_engine()
            self.df = self.df.copy()

            logger.info("Running impact simulation...")
            self.impact_metrics = simulate_impact(self.df)
            self.roi = compute_roi(self.impact_metrics)
            self.strategy_breakdown = strategy_impact_breakdown(self.df)
        except Exception:
            logger.exception("run_simulation failed")
            raise
        finally:
            logger.info(
                "run_simulation done in %.2fs rows=%s",
                time.perf_counter() - t0,
                len(self.df) if self.df is not None else 0,
            )
        return self.impact_metrics, self.roi, self.strategy_breakdown

    def run_full_pipeline(self, checkpoint_dir=None):
        """Execute the entire end-to-end flow"""
        logger.info("=== STARTING FULL PIPELINE EXECUTION ===")
        self.load_and_process()
        self.run_segmentation()
        self.run_predictive_intelligence()
        self.run_decision_engine()
        self.run_simulation()
        if checkpoint_dir:
            try:
                out = Path(checkpoint_dir)
                out.mkdir(parents=True, exist_ok=True)
                self.df.to_parquet(out / "processed.parquet", index=False)
                logger.info("Checkpoint written to %s", out / "processed.parquet")
            except Exception:
                logger.exception("Checkpoint write failed")
                raise
        logger.info("=== FULL PIPELINE EXECUTION COMPLETE ===")
        return self.df, self.impact_metrics, self.roi, self.strategy_breakdown
