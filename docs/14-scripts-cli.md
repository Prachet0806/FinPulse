# 14 — Scripts and CLI

All commands run from the repository root and accept `--log-level` unless noted.

| Command | Purpose |
|---|---|
| `python scripts/run_pipeline.py --mode train [--data-path PATH]` | Train on labeled data; publish a bundle and held-out evaluation. This is the default CLI mode. |
| `python scripts/run_pipeline.py --mode inference --data-path PATH` | Score an unlabeled cohort with the stored bundle. |
| `python scripts/score_customers.py --data-path PATH [--checkpoint-dir PATH]` | Explicit inference command; optionally save scored Parquet output. |
| `python scripts/run_predictive_intelligence.py` | Print held-out prediction samples after training. |
| `python scripts/run_decision_engine.py` | Print held-out strategy scenarios after training. |
| `python scripts/run_impact_simulation.py` | Print held-out aggregate scenario economics after training. |
| `python scripts/run_ingestion.py` | Inspect loading and validation. |
| `python scripts/run_feature_engineering.py` | Inspect legacy standalone feature generation. |
| `python scripts/run_segmentation.py` | Inspect legacy standalone clustering. |

The last three stage inspectors are demonstration utilities. Production-style training and inference use `pipeline/model_bundle.py`, which fits preprocessing only on the training cohort.
