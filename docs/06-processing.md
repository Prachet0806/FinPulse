# 06 — Processing (Feature Engineering)

Source: `processing/feature_engineering.py`

Role in pipeline: Step 2 — translate telco fields into fintech domain concepts,
derive behavioral features, one-hot encode categoricals, persist encoder columns
for inference alignment.

Entry point:

```python
from processing.feature_engineering import build_features
df = build_features(clean_df)  # save_encoder=True by default
```

## 1. `fintech_rename_columns(df)`

Maps telecom columns → fintech equivalents (idempotent):

| Source | Target |
|--------|--------|
| `tenure` | `account_tenure_months` |
| `MonthlyCharges` | `monthly_spend` |
| `TotalCharges` | `lifetime_value` |
| `Contract` | `account_plan` |
| `PaymentMethod` | `payment_behavior` |
| `InternetService` | `primary_product` |
| `TechSupport` | `priority_support` |
| `OnlineSecurity` | `fraud_protection` |
| `PaperlessBilling` | `digital_adoption` |

Details:

- Idempotency guard: if `monthly_spend` and `lifetime_value` already exist,
  skips rename/scale (safe re-runs).
- Missing source columns → warning, continue with what exists (permissive).
- Applies `FINTECH_SCALE` (10x) to `monthly_spend` and `lifetime_value`
  via `to_numeric(coerce) * FINTECH_SCALE` to model realistic fintech ARPU/CLV.

## 2. `create_behavioral_features(df)`

Derives customer-intelligence features (fail-closed on `tenure==0`):

- **`avg_revenue_per_month`** = `lifetime_value / account_tenure_months`.
  `tenure==0` or missing → `NaN` + `is_zero_tenure=1` flag (never silent 0).
  Infinities → `NaN` for downstream quarantine.
- **`engagement_score`**: weighted combo
  `monthly_spend * w1 + avg_revenue_per_month(fillna 0) * w2`
  with `ENGAGEMENT_WTS=(0.5, 0.5)`, min-max normalized to `[0, 1]`;
  degenerate range → `0.0`.
- **`tenure_segment`**: `pd.cut` on tenure with
  `TENURE_BINS=[0,12,24,48,72] + [inf]`, labels
  `new / early / mid / loyal / veteran`, `include_lowest=True`.
  `>72` never NaN by construction.

## 3. `build_features(df, save_encoder=True)`

Full pipeline:

1. `fintech_rename_columns`
2. `create_behavioral_features`
3. One-hot encode categoricals in `_CATEGORICAL_COLS`
   (`account_plan`, `payment_behavior`, `primary_product`, `tenure_segment`):
   - Other categoricals: `get_dummies(..., drop_first=True)`.
   - `tenure_segment`: `drop_first=False` so `tenure_segment_new` always
     exists — the fail-closed eligibility gate for `cashback_offer`
     depends on it.
4. Persist `list(df.columns)` to `models/encoder_columns.json`
   (`ENCODER_COLUMNS_PATH`), best-effort with `OSError` warning.

Known follow-up (documented in docstring): dummies are currently created
before train/test split; full fit-on-train/apply-to-test encoding is tracked.
Inference already reindexes to persisted columns.

## 4. `align_inference_columns(df)`

```python
from processing.feature_engineering import align_inference_columns
df_aligned = align_inference_columns(new_df)
```

Reindexes an inference frame to persisted training columns
(`fill_value=0`), so unseen categories become all-zero instead of crashing.
No-op with warning if `encoder_columns.json` is missing/unreadable.

## Outputs

New columns include:

- `account_tenure_months`, `monthly_spend`, `lifetime_value`, ...
- `avg_revenue_per_month`, `is_zero_tenure`, `engagement_score`, `tenure_segment`
- One-hot columns, critically `tenure_segment_new` (+ other tenure levels)

## Tests

- `tests/test_processing.py`
