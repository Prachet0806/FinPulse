import logging
import pandas as pd
from configs.settings import RAW_DATA_PATH, LOG_FORMAT, LOG_LEVEL

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

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Central orchestrator for FinPulse data and intelligence pipelines.
    Provides a unified API for CLI scripts and the Dashboard.
    """

    def __init__(self, data_path=RAW_DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.churn_model = None
        self.impact_metrics = None
        self.roi = None
        self.strategy_breakdown = None

    def load_and_process(self):
        """Step 1 & 2: Ingestion & Feature Engineering"""
        logger.info(f"Starting ingestion from {self.data_path}...")
        self.df = load_data(self.data_path)
        self.df = validate_data(self.df)
        
        logger.info("Starting feature engineering...")
        self.df = build_features(self.df)
        return self.df

    def run_segmentation(self, use_stored=False):
        """Step 3: Customer Segmentation"""
        if self.df is None:
            self.load_and_process()
        
        logger.info("Starting segmentation...")
        self.df = segment_customers(self.df, save_models=not use_stored)
        self.df = label_segments(self.df)
        return self.df

    def run_predictive_intelligence(self, use_stored=False):
        """Step 4: Churn Modeling, CLV, and Priority"""
        if self.df is None or "segment_id" not in self.df.columns:
            self.run_segmentation(use_stored=use_stored)
        
        logger.info("Starting predictive intelligence...")
        
        if use_stored:
            self.churn_model = load_churn_model()
            if self.churn_model is None:
                logger.warning("No stored model found. Training new model.")
                self.churn_model = train_churn_model(self.df)
        else:
            self.churn_model = train_churn_model(self.df)
            
        self.df = score_churn(self.churn_model, self.df)
        self.df = estimate_clv(self.df)
        self.df = compute_priority(self.df)
        return self.df

    def run_decision_engine(self):
        """Step 5: Strategy Optimization"""
        if self.df is None or "churn_probability" not in self.df.columns:
            self.run_predictive_intelligence()
            
        logger.info("Applying optimization strategies...")
        self.df = apply_strategies(self.df)
        return self.df

    def run_simulation(self):
        """Step 6: Impact & ROI Simulation"""
        if self.df is None or "recommended_strategy" not in self.df.columns:
            self.run_decision_engine()
            
        logger.info("Running impact simulation...")
        self.impact_metrics = simulate_impact(self.df)
        self.roi = compute_roi(self.impact_metrics)
        self.strategy_breakdown = strategy_impact_breakdown(self.df)
        return self.impact_metrics, self.roi, self.strategy_breakdown

    def run_full_pipeline(self):
        """Execute the entire end-to-end flow"""
        logger.info("=== STARTING FULL PIPELINE EXECUTION ===")
        self.load_and_process()
        self.run_segmentation()
        self.run_predictive_intelligence()
        self.run_decision_engine()
        self.run_simulation()
        logger.info("=== FULL PIPELINE EXECUTION COMPLETE ===")
        return self.df, self.impact_metrics, self.roi, self.strategy_breakdown
