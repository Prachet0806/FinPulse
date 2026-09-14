# 05 — Ingestion

Sources:

- `ingestion/loader.py` → `load_data(path)`
- `ingestion/validator.py` → `validate_data(df, return_report=False)`,
  `_quarantine_mask(df)`

Role in pipeline: Steps 1 — load raw CSV, quarantine structurally bad rows,
impute cleanable gaps, deduplicate. Permissive fail-closed: never abort on
bad rows; abort only if nothing remains (enforced by orchestrator guard).

## `load_data(path)`

```python
from ingestion.loader import load_data
df = load_data("data/raw/Telco-Customer-Churn.csv")
```

Behavior:

- Rejects empty / non-string paths with `ValueError`.
- Resolves path and requires an existing file, else `FileNotFoundError`.
- Reads with `encoding="utf-8-sig"`, `na_values=[" ", "", "NA"]`,
  `dtype={"customerID": str, "Churn": str}`, `keep_default_na=True`.
- Fatal parse errors (`ParserError`, `EmptyDataError`, `UnicodeDecodeError`)
  are logged and re-raised as `ValueError`.
- Logs shape, total nulls, duplicate count.

It does **not** validate business rules — that is `validate_data`.

## `validate_data(df, return_report=False)`

```python
from ingestion.validator import validate_data
clean = validate_data(raw)
clean, quarantine, report = validate_data(raw, return_report=True)
```

Stages:

1. **Schema check** on `REQUIRED_COLUMNS`
   (`customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Churn`).
   Missing columns are warned + recorded in `report["missing_columns"]`.
2. **Normalize `TotalCharges` early**: `astype(str).str.strip()` →
   `to_numeric(coerce)`. Coerced-null count stored as
   `report["totalcharges_coerced_nulls"]` so imputation sees the truth.
3. **Split quarantine vs. clean** via `_quarantine_mask`:
   - `customerID` missing / blank.
   - `tenure` non-numeric or `< 0`.
   - `MonthlyCharges` non-numeric or `< 0`.
   - `Churn` not in `{"Yes", "No"}`.
   - If any required column is entirely absent → all rows quarantined.
4. **Deduplicate clean** on `customerID` (`keep="first"`);
   count in `report["n_deduped"]`.
5. **Typed imputation on clean only** (no forward-fill):
   - `TotalCharges`: median + `is_totalcharges_missing` flag (0.0 if median is NaN).
   - Other numeric columns: median (except `TotalCharges`, already handled).
   - Object/categorical columns: mode, else `"Unknown"`.
6. **Coerce core numerics**: `tenure`, `MonthlyCharges`, `TotalCharges`
   → `to_numeric(coerce)`.

Return contract:

- Default: `clean_df` (backward compatible).
- `return_report=True`: `(clean_df, quarantine_df, report)` where report has
  `n_input`, `n_quarantined`, `n_deduped`, `n_output`
  (+ optional `missing_columns`, `totalcharges_coerced_nulls`).

## Quarantine output

The orchestrator writes non-empty quarantine frames to:

```text
data/processed/quarantine_ingestion_<YYYYMMDD_HHMMSS>.csv
```

via `_write_quarantine()` and logs `in / quarantined / out` counts.
Inspect these CSVs to diagnose source-data issues.

## Edge cases

| Input | Outcome |
|-------|---------|
| Missing file | `FileNotFoundError` from loader |
| Empty / unparsable CSV | `ValueError` from loader |
| Missing required column | Warning + all rows quarantined |
| Blank `customerID`, negative tenure/charges, bad `Churn` | Row quarantined |
| Duplicate `customerID` | First kept, rest counted as deduped |
| Blank `TotalCharges` (`" "`) | Coerced to NaN → median-imputed + flagged |

## Tests

- `tests/test_ingestion.py`
- `tests/test_validator_quarantine.py`
