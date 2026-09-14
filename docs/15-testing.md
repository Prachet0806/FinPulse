# 15 — Testing

Sources: `tests/` — run with `pytest`. CI runs `pytest` + `pip-audit`.

```bash
pytest
pytest tests/test_allocator.py -v
```

## Coverage map

| Test file | What it locks in |
|-----------|------------------|
| `test_ingestion.py` | Loader + validator happy paths |
| `test_validator_quarantine.py` | Structural quarantine rules, report counts |
| `test_processing.py` | Fintech rename, 10x scaling idempotency, tenure/engagement features, `tenure_segment_new` presence |
| `test_segmentation*` (via `conftest.py` fixtures) | Clustering smoke / K-selection |
| `test_churn_features_parity.py` | Train/serve `FEATURE_COLS` parity, feedback-column exclusion, unseen-category reindex |
| `test_clv_edges.py` | Fail-closed CLV (NaN churn → 0), denominator floor, non-finite quarantine flag |
| `test_eligibility_failclosed.py` | NaN/missing gating → `-inf`; missing `tenure_segment_new` disqualifies cashback; `fee_waiver` always eligible |
| `test_allocator.py` | Budget respected, caps (`max(1, ceil(N*cap))`), ROI-density order, zero budget → all `no_action` |
| `test_simulator_parity.py` | Simulator gross/net math matches optimizer; quarantine counts; input never mutated |

`conftest.py` provides shared fixtures (sample frames, model artifacts).

## Conventions for new tests

- One behavior per test; name for the rule
  (e.g. `test_zero_budget_blocks_zero_cost_strategies`).
- Build minimal DataFrames inline — do not depend on the full raw CSV.
- Assert fail-closed outcomes explicitly (`-inf`, `no_action`, `0.0`,
  `n_quarantined`).
- For economics changes, update both `decision_engine` and `simulation`
  expectations (parity requirement).
- Keep tests vectorized and fast; no network, no dashboard run.
