# 10 — Decision engine

`decision_engine/optimizer.py` estimates per-customer strategy value as assumed churn risk × estimated CLV × assumed effectiveness, minus configured cost and risk penalty. `eligibility.py` masks actions outside the demonstration rules. Invalid probabilities or CLV values contribute no gain. The fixed values in `strategy_library.py` are unvalidated scenario assumptions.

`portfolio_allocator.py` enforces one action per customer, a global monthly budget, and strategy capacity limits. For at most 200 customers, an integer solver selects the highest total estimated net value. Larger cohorts use a greedy ROI-density approximation. The allocation summary records which method was used. Zero or negative budget assigns `no_action` to everyone. A runtime check rejects overspending.

The decision output is a scenario recommendation. Credit and lending actions require real cost, risk, policy, and treatment-effect evidence before they can be used operationally. The current optimizer has no causal estimate of incremental retention.
