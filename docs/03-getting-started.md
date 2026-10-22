# 03 — Getting started

Install Python 3.10 or newer and dependencies from `requirements.txt`. From the repository root:

```bash
python -m venv .venv
pip install -r requirements.txt
python scripts/run_pipeline.py --mode train
```

Training expects a labeled CSV. The default is the bundled Telco proxy. It writes `models/finpulse_bundle.joblib` and `data/processed/evaluation.parquet`; both paths are ignored by Git. The evaluation file contains only held-out rows. Set `--data-path PATH` to select another labeled dataset.

Score an unlabeled cohort with the stored bundle:

```bash
python scripts/score_customers.py --data-path path/to/customers.csv
```

The inference CSV needs `customerID`, `tenure`, `MonthlyCharges`, and `TotalCharges`; it need not include `Churn`. A missing or incompatible bundle fails explicitly. Use `--checkpoint-dir PATH` to save scored rows.

For the dashboard, set `FINPULSE_DASHBOARD_TOKEN` to a secret value and run `streamlit run dashboard/app.py`. Enter that token in the sidebar. The dashboard reads the saved held-out evaluation and never trains a model. A shared token is a minimum local guard; production access requires a proper authentication and authorization layer.

Run checks with `pytest -q`. The data, bundle, and quarantine output are local files and should be handled under an approved privacy and retention policy when using non-demo records.
