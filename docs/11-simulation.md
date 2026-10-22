# 11 — Scenario simulation

`simulation/impact_simulator.py:simulate_impact` calculates assumed gross retention gain, strategy cost, net benefit, risk penalty, and risk-adjusted net for assigned actions. Inputs with unknown strategies, invalid churn probabilities, or invalid CLV are counted as quarantined and excluded. It does not mutate the input frame.

`compute_roi` reports gross `(gain − cost) / cost` when cost is positive; zero cost with positive gain yields infinity. The dashboard separately computes risk-adjusted ROI from risk-adjusted net and cost. `strategy_impact_breakdown` reports action counts and unit costs.

All gain and ROI figures are scenario estimates. Effectiveness, cost, prediction horizon, and the relationship between estimated CLV and intervention benefit remain unvalidated for real customers. Use measured outcomes and comparison groups before making realized-impact claims.
