# 10 — Decision Engine

Sources (`decision_engine/`):

- `strategy_library.py` → `STRATEGIES`
- `risk_adjustment.py` → `calculate_risk_penalty`
- `eligibility.py` → `apply_eligibility_mask`
- `portfolio_allocator.py` → `allocate_portfolio`
- `optimizer.py` → `optimize_strategies_vectorized`
- `apply_strategies.py` → `apply_strategies` (thin wrapper)

Role in pipeline: Step 5 — 4-layer V2 enterprise framework that turns
risk/value into a single `recommended_strategy` per customer.

## Layer overview

```python
from decision_engine.apply_strategies import apply_strategies
df = apply_strategies(df)  # adds recommended_strategy
```

`optimize_strategies_vectorized(df)` executes:

1. **Base expected value**: `(churn × CLV × effectiveness) − cost`.
2. **Risk adjustment**: subtract probability-scaled penalty.
3. **Eligibility mask**: fail-closed business rules → `-inf` if ineligible.
4. **Portfolio allocation**: greedy ROI-density under budget + caps.

Fail-closed numeric views: NaN churn → `0.0` gain in the optimizer
(no phantom gain); NaN CLV → `0.0`. Risk layer separately treats NaN
churn as max penalty (see below).

## Strategy library

`STRATEGIES` (static estimates — need A/B validation):

| Strategy | Cost | Effectiveness | Risk penalty | Cap |
|----------|------|---------------|--------------|-----|
| `cashback_offer` | 200 | 0.25 | 0.0 | 20% |
| `fee_waiver` | 100 | 0.20 | 20.0 | 30% |
| `credit_limit_increase` | 0 | 0.18 | 50.0 | 15% |
| `loan_offer` | 50 | 0.22 | 10.0 | 10% |
| `no_action` | 0 | 0.0 | 0.0 | 100% (uncapped fallback) |

`no_action` baseline value is `0.0` and is assigned when nothing else has
positive net value, budget is exhausted, caps are hit, or rules disqualify.

## Layer 2 — Risk adjustment

```python
calculate_risk_penalty(df, strategy, params, use_clv_scaling=False)
```

- NaN churn → `1.0` (max penalty, fail-closed).
- `credit_limit_increase`: `50 + churn × 200` (`RISK_SLOPES`).
- `loan_offer`: `10 + churn × 100`.
- All others: flat `risk_penalty` (e.g. `fee_waiver → 20.0`,
  `cashback_offer → 0.0`).
- Optional CLV-scaled mode (`use_clv_scaling=True`, default off):
  `base + churn × slope × (1 + CLV/1000)`, logs flat vs. scaled means.

## Layer 3 — Eligibility (fail-closed)

```python
apply_eligibility_mask(df, strategy, net_value_series)
```

Missing/NaN gating values → ineligible (`-inf`), never eligible.

| Strategy | Rule |
|----------|------|
| `credit_limit_increase` | Requires `account_tenure_months > 12` and `churn_probability ≤ 0.6`; NaN tenure/churn → ineligible |
| `loan_offer` | Requires `estimated_clv ≥ 500`; NaN CLV → ineligible |
| `cashback_offer` | Excludes `tenure_segment_new == 1` (anti-gamification); missing `tenure_segment_new` column → ineligible for **all** rows + quarantine warning; NaN flag → ineligible |
| `fee_waiver` / `no_action` | Always eligible by design |

Missing gating column entirely → quarantine warning + NaN series →
ineligible for that strategy.

## Layer 4 — Portfolio allocation

```python
allocate_portfolio(df, strategy_net_values, strategy_costs)
# returns (df, allocation_summary)
```

Greedy ROI-density (knapsack approximation; LP solver is roadmap):

1. Flatten to `(customer, strategy, net_value, cost, ROI)` rows for every
   strategy with `net_value > EPS (1e-9)`. `ROI = net/cost`, `inf` if cost 0.
2. Sort by `ROI` desc, tie-break `net_value` desc.
3. Greedily assign each customer's best remaining opportunity if:
   - Customer unassigned so far (one strategy per customer).
   - `budget_remaining >= cost` (else skip, continue scanning).
   - `assigned[strategy] < max(1, ceil(N × cap_percent))`.
4. Assert `total_cost <= MONTHLY_BUDGET + EPS` (overspend guard).

Special cases:

- `MONTHLY_BUDGET <= 0` → all `no_action`, even zero-cost strategies.
- No positive-ROI candidates → all `no_action` + warning.
- Duplicate index → reset + warning (so `df.at` is unambiguous).

Summary:

```python
{
  "total_net_value": float,
  "total_cost": float,
  "budget_remaining": float,
  "assigned_counts": {strategy: int, ..., "no_action": int},
}
```

## Outputs

- `recommended_strategy` per row (one of `STRATEGIES` keys).
- Allocation summary logged by the optimizer (`Allocation summary: ...`).

## Tests

- `tests/test_eligibility_failclosed.py`
- `tests/test_allocator.py`
