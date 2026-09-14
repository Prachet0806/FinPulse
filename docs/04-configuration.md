# 04 — Configuration

Source: `configs/settings.py`

Centralizes paths, model hyper-parameters, decision constraints, and the
"magic numbers" used across the pipeline. This is the single source of truth —
do not hardcode these values elsewhere.

## Paths

| Constant | Default | Notes |
|----------|---------|-------|
| `BASE_DIR` | repo root (`Path(__file__).resolve().parent.parent`) | All other paths derive from this |
| `RAW_DATA_PATH` | `BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"` | Overridable via `FINPULSE_RAW_PATH` env var, resolved to absolute path |
| `MODELS_DIR` | `BASE_DIR / "models"` | Scalers, KMeans, churn model, metrics, encoder/feature columns |
| `PROCESSED_DIR` | `BASE_DIR / "data" / "processed"` | Quarantine CSVs |

`ensure_dirs()` creates `MODELS_DIR` and `PROCESSED_DIR` if missing.
Call from entrypoints (`scripts/*`, dashboard), not library code.

## Model parameters

| Constant | Value | Used by |
|----------|-------|---------|
| `RANDOM_SEED` | `42` | KMeans, train/test split, RandomForest, dashboard sampling |
| `TEST_SIZE` | `0.2` | `train_churn_model` stratified split |
| `DISCOUNT_RATE` | `0.1` | CLV denominator `1 + Discount − Retention` |
| `DEFAULT_CLUSTERS` | `4` | Fallback when `auto_k=False` |

## Decision constraints

| Constant | Value | Used by |
|----------|-------|---------|
| `MONTHLY_BUDGET` | `15000` | `allocate_portfolio` global budget; `<= 0` forces all `no_action` |

## Centralized magic numbers

| Constant | Value | Meaning |
|----------|-------|---------|
| `FINTECH_SCALE` | `10` | 10x multiplier on `monthly_spend` / `lifetime_value` |
| `ENGAGEMENT_WTS` | `(0.5, 0.5)` | Weights for `monthly_spend` vs `avg_revenue_per_month` in engagement score |
| `TENURE_BINS` | `[0, 12, 24, 48, 72]` | Cut points for `new/early/mid/loyal/veteran` (+ `inf` so `>72` never NaN) |
| `CLV_MIN_DENOM` | `0.05` | Floor for CLV denominator |
| `EPS` | `1e-9` | Positive-ROI threshold (`net_value > EPS`); budget overspend tolerance |
| `RISK_SLOPES` | `{"credit_limit_increase": 200, "loan_offer": 100}` | Churn-scaled penalty slopes |

## Ingestion schema

```python
REQUIRED_COLUMNS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
```

Used by `ingestion/validator.py`. Missing column → entire frame quarantined
for that check; see `05-ingestion.md`.

## Logging

- `LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- `LOG_LEVEL` from `FINPULSE_LOG_LEVEL` env var, default `"INFO"`.
- `setup_logging(level=None)` configures root logging once via `dictConfig`
  with a single console handler. Call only from entrypoints
  (`scripts/*`, `dashboard/app.py`).

## How to change settings safely

1. Prefer env vars (`FINPULSE_RAW_PATH`, `FINPULSE_LOG_LEVEL`) for paths/verbosity.
2. For budget, discount rate, slopes, bins — edit `configs/settings.py`
   directly; all consumers read from there.
3. Changing `FINTECH_SCALE` or `TENURE_BINS` alters features and model
   inputs — retrain (`save_models=True`) and refresh persisted artifacts.
