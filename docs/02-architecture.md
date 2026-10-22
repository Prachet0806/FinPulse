# 02 — Architecture

```text
Labeled CSV → validation → train/holdout split
                 ├─ train: fit FeatureTransformer → fit clusters → fit churn model
                 └─ holdout: transform → cluster → score → CLV → allocation → simulation
                           ↓
                versioned model bundle + held-out evaluation snapshot
                           ↓
                 access-gated Streamlit dashboard

Unlabeled CSV → validation → load bundle → fitted transform → cluster → score
             → CLV → allocation → scenario simulation → optional Parquet output
```

The training path is implemented in `pipeline/model_bundle.py`. `pipeline/orchestrator.py` selects training or inference explicitly. Fitted categorical levels, imputation values, and engagement normalization bounds live in the bundle with the segmentation and churn models. The dashboard checks the evaluation snapshot's run ID against the bundle.

The allocator solves small cohorts exactly (up to 200 customers) and applies an ROI-density heuristic for larger cohorts. Silhouette search samples up to 2,000 rows. Local files under `models/` and `data/processed/` may contain sensitive information when real records are used.
