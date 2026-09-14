# 14 — Scripts & CLI

Sources: `scripts/`

Thin CLI entrypoints over `PipelineOrchestrator`. Every script:

- Appends project root to `sys.path` for local execution.
- Calls `setup_logging(args.log_level)` + `ensure_dirs()`.
- Accepts `--data-path` (defaults to canonical `RAW_DATA_PATH`) and
  `--log-level`; exits non-zero (`sys.exit(1)`) on fatal errors.
- Requires `PYTHONPATH="."` (or the `sys.path` bootstrap) to resolve
  top-level packages (`configs`, `pipeline`, ...).

## `run_pipeline.py` — full flow

```bash
python scripts/run_pipeline.py [--data-path PATH] [--log-level LEVEL]
```

Runs `run_full_pipeline()`, prints sample of
`segment_label, churn_probability, estimated_clv, recommended_strategy`.

## Stage scripts

| Script | Calls |
|--------|-------|
| `run_ingestion.py` | `load_and_process()` (load + validate + features) |
| `run_feature_engineering.py` | `load_and_process()` |
| `run_segmentation.py` | `run_segmentation()` |
| `run_predictive_intelligence.py` | `run_predictive_intelligence()` |
| `run_decision_engine.py` | `run_decision_engine()` |
| `run_impact_simulation.py` | `run_simulation()` |

Each prints/inspects its stage output; see individual script headers for
exact flags (all share `--data-path` / `--log-level`).

## Common patterns

```bash
$env:PYTHONPATH="."  # PowerShell
python scripts/run_pipeline.py --data-path "data/raw/Telco-Customer-Churn.csv" --log-level INFO
```

```bash
export PYTHONPATH="."  # Unix/macOS
python scripts/run_pipeline.py
```

Override data path without flags:

```bash
$env:FINPULSE_RAW_PATH="data/raw/custom.csv"
python scripts/run_pipeline.py
```
