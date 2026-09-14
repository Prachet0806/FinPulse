# 12 — Pipeline Orchestration

Source: `pipeline/orchestrator.py` → `PipelineOrchestrator`

Unified API for CLI scripts and the dashboard. Permissive fail-closed:
stages log + quarantine + continue; abort only if the frame is empty after
quarantine (`_guard_nonempty` → `ValueError`).

## State

```python
from pipeline.orchestrator import PipelineOrchestrator
orc = PipelineOrchestrator(data_path=RAW_DATA_PATH)
orc.df               # working frame
orc.churn_model      # trained / loaded model
orc.impact_metrics   # simulate_impact dict
orc.roi              # compute_roi float
orc.strategy_breakdown  # DataFrame
```

## Stage methods

Each stage is timed, logs duration + row count, copies frames defensively,
and auto-runs prerequisites if state is missing.

### `load_and_process()` — Steps 1–2

1. `load_data(self.data_path)`
2. `validate_data(raw, return_report=True)` → log report.
3. `_write_quarantine(quarantine, "ingestion")` →
   `data/processed/quarantine_ingestion_<ts>.csv` (no-op if empty).
4. `_guard_nonempty("ingestion")`
5. `build_features(clean)`

### `run_segmentation(use_stored=False)` — Step 3

Requires `self.df` (else `load_and_process()`).

1. `segment_customers(df, save_models=not use_stored, use_stored=use_stored)`
2. `label_segments(df)`

### `run_predictive_intelligence(use_stored=False)` — Step 4

Requires `segment_id` (else `run_segmentation()`).

1. `load_churn_model()` if `use_stored` (fallback: train + warn if missing),
   else `train_churn_model(df)`.
2. `score_churn(model, df)` → `churn_probability`
3. `estimate_clv(df)` → `estimated_clv`
4. `compute_priority(df)` → `priority_score`

### `run_decision_engine()` — Step 5

Requires `churn_probability` (else `run_predictive_intelligence()`).

- `apply_strategies(df)` → `recommended_strategy`

### `run_simulation()` — Step 6

Requires `recommended_strategy` (else `run_decision_engine()`).

- `simulate_impact(df)` → `impact_metrics`
- `compute_roi(impact)` → `roi`
- `strategy_impact_breakdown(df)` → `strategy_breakdown`

### `run_full_pipeline(checkpoint_dir=None)`

Runs all stages in order. If `checkpoint_dir` is given, writes
`processed.parquet` there (creates dirs; requires `pyarrow`).

```python
df, impact, roi, breakdown = PipelineOrchestrator().run_full_pipeline()
df, impact, roi, breakdown = PipelineOrchestrator().run_full_pipeline(
    checkpoint_dir="data/processed/checkpoint"
)
```

## Helpers

- `_write_quarantine(df, stage)`: timestamped CSV write, `OSError`-safe.
- `_guard_nonempty(stage)`: raises `ValueError` on empty/None frame.

## Error handling

Every stage wraps in try/except → logs with traceback → re-raises.
`finally` always logs elapsed time + row count. CLI scripts exit non-zero
on fatal errors; dashboard shows a ref-ID error card.
