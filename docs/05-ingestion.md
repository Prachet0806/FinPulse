# 05 — Ingestion and validation

`ingestion/loader.py:load_data` reads a CSV. `ingestion/validator.py:validate_data` separates clean and quarantined rows and returns counts when `return_report=True`.

Training requires `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn` (`Yes` or `No`). Inference sets `require_target=False` and does not require `Churn`. The full workflow sets `impute=False`, then fits imputation on training rows only in `FeatureTransformer`.

Rows with missing IDs, invalid churn labels in training, negative or non-finite core numeric values, or conflicting records for the same customer ID are quarantined. Exact duplicate records are reduced to one and counted. Missing or unparseable `TotalCharges` is flagged and later filled using the training median. If no clean rows remain, the workflow fails.

The orchestrator writes quarantined rows to timestamped CSV files under `data/processed/`. These files may contain all original customer fields and require appropriate access and retention controls.
