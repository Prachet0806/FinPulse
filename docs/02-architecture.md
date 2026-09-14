# 02 — Architecture

## Layer diagram

```text
[Raw Data] --> (Ingestion & Validation + quarantine)
                        ↓
[Feature Engineering] --> (Domain Transformation, Categorical Encoding)
                        ↓
[Customer Intelligence Layer]
  ├── Segmentation Engine (Clustering + metadata)
  ├── Predictive Engine (Churn Model + feature parity)
  └── Valuation Engine (CLV Calculation, fail-closed)
                        ↓
[Decision Engine]
  ├── Risk Adjustment (Probability-scaled Penalties)
  ├── Eligibility Masking (Fail-closed Tenure & Risk Rules)
  └── Portfolio Allocator (ROI Density vs. Global Budget + summary)
                        ↓
[Impact Simulation] --> (Gross + Risk-Adjusted Net, ROI & Breakdown)
                        ↓
[Executive Dashboard] --> (Interactive UI & Reporting)
```

## Module map

| Layer | Code | Docs |
|-------|------|------|
| Config | `configs/settings.py` | `04-configuration.md` |
| Ingestion | `ingestion/loader.py`, `ingestion/validator.py` | `05-ingestion.md` |
| Processing | `processing/feature_engineering.py` | `06-processing.md` |
| Segmentation | `intelligence/segmentation/` | `07-intelligence-segmentation.md` |
| Churn | `intelligence/churn/` | `08-intelligence-churn.md` |
| CLV / Priority | `intelligence/clv/`, `intelligence/customer_priority.py` | `09-intelligence-clv-priority.md` |
| Decisions | `decision_engine/` | `10-decision-engine.md` |
| Simulation | `simulation/` | `11-simulation.md` |
| Orchestration | `pipeline/orchestrator.py` | `12-pipeline.md` |
| UI | `dashboard/app.py` | `13-dashboard.md` |
| Entrypoints | `scripts/` | `14-scripts-cli.md` |

`recommendations/` and `reporting/` exist as placeholder packages.

## End-to-end workflow

1. **Ingestion**: canonical path `configs/settings.py: RAW_DATA_PATH`,
   overridable with `FINPULSE_RAW_PATH`. Bad rows quarantined to
   `data/processed/quarantine_*.csv`; pipeline continues on clean rows,
   aborts only if nothing remains.
2. **Domain transformation**: telco → fintech fields, idempotent 10x scaling,
   `tenure==0 → NaN + is_zero_tenure`, min-max engagement score,
   tenure bands, one-hot encoding with persisted columns.
3. **Segmentation**: scale + cluster into actionable segments.
4. **Prediction**: RandomForest churn scoring + CLV + priority.
5. **Decision optimization**: evaluate strategies, mask ineligible,
   allocate by ROI density under budget/caps.
6. **Impact simulation**: gross gain, cost, risk total, risk-adjusted net, ROI.
7. **Reporting**: Streamlit/Plotly dashboard; optional
   `processed.parquet` checkpoint for lineage.

## Core design principles

- **ROI-driven optimization**: optimize the preventative response,
  not just churn probability.
- **Permissive pipeline, fail-closed decisions**:
  pipeline quarantines + continues; decisions treat missing/NaN as
  ineligible / `no_action`, zero budget blocks even zero-cost actions.
- **Explainability**: segment labels + named strategies + breakdown tables.
- **Scalability**: vectorized processing/decision layers, no distributed
  compute required; dashboard downsamples + uses WebGL past 5k rows.
- **Train/serve parity**: persisted `encoder_columns.json`,
  `churn_feature_columns.json`, scalers/models; inference reindexes,
  unseen categories map to 0.

## Key artifacts

- `models/segment_scaler.joblib`, `segment_kmeans.joblib`,
  `segmentation_metadata.json`
- `models/churn_model.joblib`, `models/churn_metrics.json`,
  `models/churn_feature_columns.json`
- `models/encoder_columns.json`
- `data/processed/quarantine_*.csv`
- Optional checkpoint: `<checkpoint_dir>/processed.parquet`
