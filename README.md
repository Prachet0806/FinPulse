# FinPulse

FinPulse is a **demonstration decision-support system** for customer retention. It trains a churn model on labeled historical records, scores a held-out cohort, and estimates strategy scenarios under a fixed budget. The bundled Telco churn dataset is a proxy; the strategy effects, costs, credit risk penalties, 10× financial scaling, and projected returns have **not** been validated for a financial institution. Do not treat its scenarios as measured gains or authorize real customer actions from them.

## Current workflow

1. Training loads a labeled CSV, quarantines unusable and conflicting records, and splits rows into train and holdout cohorts **before fitting preprocessing**.
2. A fitted feature transformer stores imputation values, engagement normalization bounds, and categorical levels. Clustering and the churn model fit on training rows only. Segment labels remain neutral because KMeans IDs have no inherent business meaning.
3. The model bundle is published to `models/finpulse_bundle.joblib`. A separate `data/processed/evaluation.parquet` contains predictions for the untouched holdout. The dashboard reads this snapshot; opening it does not retrain.
4. Inference accepts an unlabeled CSV and uses the stored bundle without fitting. A missing or incompatible bundle is an error.
5. The decision engine estimates gain, cost, and risk penalties from static scenario assumptions. Allocation uses an exact integer solution for at most 200 customers and a greedy approximation above that size.

## Setup

Python **3.10+** is required. CI runs Python 3.11.

```bash
python -m venv .venv
pip install -r requirements.txt
python scripts/run_pipeline.py --mode train
```

The training command uses `data/raw/Telco-Customer-Churn.csv` by default. It writes the bundle and holdout snapshot under ignored local directories. `--data-path PATH` selects another labeled file. The input requires `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn` (`Yes` or `No`).

To score new customers after training:

```bash
python scripts/score_customers.py --data-path path/to/unlabeled.csv
```

Inference requires the same four non-target fields. `--checkpoint-dir PATH` optionally writes the scored cohort as Parquet. The default full pipeline API mode is inference; training must be requested with `mode="train"`.

To open the holdout dashboard, set a secret access token and then start Streamlit:

```powershell
$env:FINPULSE_DASHBOARD_TOKEN = "choose-a-long-random-secret"
streamlit run dashboard/app.py
```

The application checks a shared token before showing customer-level records. A real deployment still needs an authenticated gateway, individual access controls, transport security, and an approved data-retention policy. The dashboard refuses to load a holdout snapshot whose run ID differs from the active bundle.

## Economic interpretation

`total_retention_gain = churn_probability × estimated_clv × assumed_effectiveness` is a scenario calculation, not an experimentally measured treatment effect. `net_benefit` subtracts assumed strategy cost; `net_risk_adjusted` additionally subtracts the configured risk penalty. The dashboard names these as scenario estimates and exposes the assumptions. Prediction horizon, treatment effectiveness, cost of capital, and real-world calibration require representative dated data and outcome measurement before production use.

The bundled data are cross-sectional, so the 20% stratified holdout provides offline proxy evidence only. It does not establish temporal performance or deployment readiness.

## Quality and limits

Run `pytest -q` for local checks. CI runs the test suite and `pip-audit` as required gates. As of October 2026, Streamlit 1.64 requires Pillow below 13 while published Pillow advisories require 13 for fixes; the strict dependency gate will remain red until a compatible release or dashboard migration is available. The allocator is exact only for cohorts of at most 200 customers; larger cohorts use a heuristic. K selection samples up to 2,000 rows for silhouette scoring. No million-row performance claim is made without a benchmark.

See [the documentation index](docs/README.md) for module details.
