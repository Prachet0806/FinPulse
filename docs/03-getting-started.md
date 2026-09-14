# 03 — Getting Started

## Prerequisites

- Python 3.9+
- `pip install -r requirements.txt`
- Project root as `PYTHONPATH` for `scripts/` runs.

Key dependencies (`requirements.txt`):

- `pandas`, `numpy`, `scikit-learn`, `joblib`, `pyarrow`
- `streamlit`, `plotly`
- `pytest`, `matplotlib`, `seaborn`, `pyyaml`

## 1. Installation

```bash
git clone https://github.com/Prachet0806/FinPulse.git
cd FinPulse
pip install -r requirements.txt
```

## 2. Run the full pipeline

```bash
# Windows (PowerShell)
$env:PYTHONPATH="."
python scripts/run_pipeline.py
python scripts/run_pipeline.py --data-path "data/raw/Telco-Customer-Churn.csv" --log-level INFO

# Unix / macOS
export PYTHONPATH="."
python scripts/run_pipeline.py
```

All `scripts/` entrypoints accept `--data-path` (defaults to canonical
`RAW_DATA_PATH`) and `--log-level`, and exit non-zero on fatal errors.

Expected output: sample table with
`segment_label`, `churn_probability`, `estimated_clv`, `recommended_strategy`.

## 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

## 4. Run individual stages

```bash
python scripts/run_ingestion.py
python scripts/run_feature_engineering.py
python scripts/run_segmentation.py
python scripts/run_predictive_intelligence.py
python scripts/run_decision_engine.py
python scripts/run_impact_simulation.py
```

Each script is a thin CLI over `PipelineOrchestrator` (see `12-pipeline.md`
and `14-scripts-cli.md`).

## Environment overrides

| Variable | Effect |
|----------|--------|
| `FINPULSE_RAW_PATH` | Override `RAW_DATA_PATH` |
| `FINPULSE_LOG_LEVEL` | Override default log level (`INFO`) |
| `PYTHONPATH="."` | Required so `scripts/` can import top-level packages |

## Troubleshooting

- `Dataset not found`: check `--data-path` / `FINPULSE_RAW_PATH`.
- `Pipeline aborted: empty dataframe after ingestion quarantine`:
  all rows failed structural validation — inspect
  `data/processed/quarantine_ingestion_*.csv`.
- `No stored model found`: `use_stored=True` path falls back to training.
- Dashboard `Pipeline failed (ref ID ...)`: check logs for the ref ID;
  usually a schema break in pipeline output.
