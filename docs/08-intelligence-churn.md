# 08 — Intelligence: Churn

Sources (`intelligence/churn/`):

- `churn_model.py` → `train_churn_model`, `load_churn_model`,
  `FEATURE_COLS`, `FEATURE_COLS_PATH`, `MODEL_PATH`, `METRICS_PATH`
- `churn_scoring.py` → `score_churn`, `_resolve_feature_cols`

Role in pipeline: Step 4a — estimate per-customer churn probability.
Fail-closed: feedback columns are excluded; invalid targets quarantined;
inference reindexes to training columns.

## Feature frame (`_build_feature_frame`)

- Target: `Churn` (case-insensitive fallback); missing → `ValueError`.
- `X` = numeric dtypes only (`int64/float64/uint8/int32/bool`).
- Drops:
  - Target column.
  - Feedback-loop columns (never features):
    `churn_probability`, `estimated_clv`, `priority_score`,
    `retention_gain`, `recommended_strategy`.
  - ID/label columns: `customerID`, `segment_id`, `segment_label`.
- Casts `bool` → `int64` for sklearn stability.

## `train_churn_model(df, save_model=True)`

```python
from intelligence.churn.churn_model import train_churn_model
model = train_churn_model(df)
```

Steps:

1. Map target `{"Yes": 1, "No": 0, 1: 1, 0: 0}`; drop rows with unmappable
   target (warning with count).
2. Set global `FEATURE_COLS = list(X.columns)` — the single shared constant
   for train/serve parity.
3. Stratified split (`TEST_SIZE=0.2`, `RANDOM_SEED=42`).
4. `RandomForestClassifier(random_state=42, class_weight="balanced_subsample",
   max_depth=12, min_samples_leaf=5)` → `fit`.
5. Metrics at `0.5` threshold: `accuracy`, `precision`, `recall`, `f1`,
   `roc_auc` (0.0 if single-class test set), `confusion_matrix`,
   `n_features`, `feature_columns`.
6. Atomic persist (`save_model=True`):
   - `models/churn_model.joblib` (temp + `os.replace`).
   - `models/churn_metrics.json`.
   - `models/churn_feature_columns.json`.
   - Attach `model.feature_names_in_ = FEATURE_COLS` for in-process parity.

## `load_churn_model()`

Loads `churn_model.joblib` if present, restores global `FEATURE_COLS` from
`churn_feature_columns.json` (fallback: `model.feature_names_in_`).
Returns `None` if no model file exists — orchestrator then trains fresh
when `use_stored=True`.

## `score_churn(model, df)`

```python
from intelligence.churn.churn_scoring import score_churn
df = score_churn(model, df)  # adds churn_probability
```

- Resolves columns via `_resolve_feature_cols` priority:
  1. In-process `FEATURE_COLS` from `churn_model` module.
  2. `model.feature_names_in_`.
  3. `models/churn_feature_columns.json`.
  4. Numeric-dtype fallback (minus target/ID/feedback cols) + warning.
- `df.reindex(columns=cols, fill_value=0)` — unseen categories → 0.
- Coerce `bool` → `int64`, leftover `object` → `to_numeric(coerce).fillna(0)`.
- `model.predict_proba(X)[:, 1]` → `churn_probability`.

## Contract with downstream

- `churn_probability` ∈ `[0, 1]` per row (float).
- Missing/NaN churn downstream is treated fail-closed:
  CLV → retention 0, optimizer → no phantom gain, risk → max penalty.

## Tests

- `tests/test_churn_features_parity.py`
