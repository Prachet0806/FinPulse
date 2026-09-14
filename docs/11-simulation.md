# 11 — Simulation

Sources (`simulation/`):

- `impact_simulator.py` → `simulate_impact`
- `roi_calculator.py` → `compute_roi`
- `strategy_breakdown.py` → `strategy_impact_breakdown`

Role in pipeline: Step 6 — quantify financial impact of `recommended_strategy`
without mutating the input frame. Reports both gross and risk-adjusted views
so the dashboard can switch between them.

## `simulate_impact(df)`

```python
from simulation.impact_simulator import simulate_impact
impact = simulate_impact(df)
```

Formula per retained row:

```text
retention_gain = churn_probability × estimated_clv × effectiveness[strategy]
total_gain   = Σ retention_gain
total_cost   = Σ cost[strategy]
net_benefit  = total_gain − total_cost
risk_total   = Σ calculate_risk_penalty(...) per strategy (parity with optimizer)
net_risk_adjusted = total_gain − total_cost − risk_total
```

Quarantine (warned, excluded from totals, never silently undercounted):

- `recommended_strategy` missing → all-zero result with
  `n_quarantined = len(df)`.
- Unknown strategy, NaN churn, or NaN CLV → row excluded;
  `n_quarantined` counts them.
- Empty after quarantine → all-zero result, preserves `n_quarantined`.

Return dict:

```python
{
  "total_retention_gain": float,
  "total_strategy_cost": float,
  "net_benefit": float,
  "risk_penalty_total": float,
  "net_risk_adjusted": float,
  "n_quarantined": int,
}
```

`risk_total` uses the same `calculate_risk_penalty` as the optimizer,
so simulator and optimizer economics match.

## `compute_roi(impact)`

```python
from simulation.roi_calculator import compute_roi
roi = compute_roi(impact)
```

Rules:

- Non-dict impact → `0.0` + warning.
- Non-numeric / NaN / negative cost → `0.0` + warning.
- `cost == 0`, `gain > 0` → `inf`.
- `cost == 0`, `gain == 0` → `0.0`.
- Else `net_benefit / cost` (`net_benefit` key preferred, else `gain − cost`);
  non-finite → `0.0`.

`gain` defaults to `total_retention_gain`, falling back to `net_benefit`.

## `strategy_impact_breakdown(df)`

```python
from simulation.strategy_breakdown import strategy_impact_breakdown
breakdown = strategy_impact_breakdown(df)  # DataFrame
```

Returns a DataFrame with `strategy | count | percent | cost_per_strategy`
(one row per assigned strategy, costs from `STRATEGIES`).
Non-DataFrame or missing `recommended_strategy` → empty DataFrame with those
columns.

## Tests

- `tests/test_simulator_parity.py`
