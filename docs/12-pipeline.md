# 12 — Pipeline orchestration

`pipeline/orchestrator.py` exposes `PipelineOrchestrator(data_path).run_full_pipeline(checkpoint_dir=None, mode="inference")`. The mode must be explicit for training: `mode="train"`.

## Training and evaluation

`pipeline/model_bundle.py:train_and_evaluate` validates labeled rows, quarantines conflicts and invalid values, splits into 80% training and 20% holdout, fits the feature transformer and segmentation on training rows, fits the churn model on training rows, and scores only the holdout. It publishes one versioned bundle and a held-out evaluation Parquet file with a matching run ID. The dashboard uses those two files.

## Inference

`score_new_customers` validates an unlabeled cohort, loads the bundle, applies fitted preprocessing and segmentation, scores churn, estimates CLV and priority, allocates strategies, and calculates scenario economics. It neither fits nor silently falls back to training. Incompatible or absent artifacts fail.

`run_full_pipeline(checkpoint_dir=...)` can write a scored Parquet checkpoint. Invalid input rows are written to timestamped quarantine CSV files under `data/processed/`; those files may contain customer information.

The pipeline returns `(scored_df, impact_dict, gross_roi, strategy_breakdown)`. `gross_roi` is based on the assumed gross benefit, while the impact dict also includes risk-adjusted net. These are unvalidated scenarios.
