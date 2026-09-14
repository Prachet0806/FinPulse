# 09 — Intelligence: CLV & Priority

Sources:

- `intelligence/clv/clv_model.py` → `estimate_clv`
- `intelligence/customer_priority.py` → `compute_priority`

Role in pipeline: Step 4b–4c — convert risk + revenue into dollar value
(CLV) and a single triage score (priority). Both are fail-closed and
vectorized.

## `estimate_clv(df)`

Financial retention formula:

```text
CLV = (ARPU * Retention) / (1 + Discount - Retention)
```

where `Retention = 1 − churn_probability`, `Discount = DISCOUNT_RATE (0.1)`.

```python
from intelligence.clv.clv_model import estimate_clv
df = estimate_clv(df)  # adds estimated_clv, is_clv_quarantined
```

Rules:

- **Fail-closed churn**: `churn_probability` coerced numeric, NaN → `1.0`
  (unknown risk → retention 0 → CLV ≈ 0).
- **Retention** clipped to `[0, 1]`.
- **ARPU** = `avg_revenue_per_month`, coerced numeric, NaN → `0.0`.
- **Denominator** `(1 + Discount − Retention)` floored at `CLV_MIN_DENOM (0.05)`.
- **Non-finite quarantine**: `inf`/`NaN` results → `0.0` +
  `is_clv_quarantined=1` flag + warning with count.

Output columns: `estimated_clv` (float), `is_clv_quarantined` (0/1).

## `compute_priority(df)`

```python
from intelligence.customer_priority import compute_priority
df = compute_priority(df)  # adds priority_score
```

- `priority_score = churn_probability × estimated_clv`.
- Both inputs coerced numeric, NaN → `0.0` (fail-closed).
- Right-skew (`skew > 5`) is logged but kept unnormalized for dashboard
  stability (size encoding in the risk matrix).

## Worked intuition

| churn | CLV | Priority | Meaning |
|-------|-----|----------|---------|
| 0.9 | $3,000 | 2,700 | High-value at risk — intervene first |
| 0.9 | $100 | 90 | Risky but low value — cheap action only |
| 0.1 | $3,000 | 300 | Valuable but safe — monitor |

The decision engine multiplies the same `churn × CLV × effectiveness`
term when computing base expected value, so priority ordering and
optimizer economics are consistent.

## Tests

- `tests/test_clv_edges.py`
