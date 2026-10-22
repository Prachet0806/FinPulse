# 04 — Configuration

`configs/settings.py` defines paths and demonstration constants. `FINPULSE_RAW_PATH` overrides the default raw CSV for the training CLI; inference commands require an explicit data path. `FINPULSE_LOG_LEVEL` controls logging. `FINPULSE_DASHBOARD_TOKEN` is required before the dashboard serves customer-level data.

| Setting | Current value | Meaning |
|---|---:|---|
| `TEST_SIZE` | `0.2` | Stratified offline holdout fraction |
| `RANDOM_SEED` | `42` | Reproducible split and models |
| `MONTHLY_BUDGET` | `15000` | Scenario allocation budget |
| `FINTECH_SCALE` | `10` | Proxy-data scaling assumption, not observed banking economics |
| `DISCOUNT_RATE` | `0.1` | CLV scenario discount assumption |
| `CLV_MIN_DENOM` | `0.05` | Numerical floor in CLV formula |

`models/finpulse_bundle.joblib` is the active local model bundle. `data/processed/evaluation.parquet` holds the corresponding scored holdout. Both are ignored by Git. A real deployment needs governed model promotion, storage permissions, transport security, and data retention.
