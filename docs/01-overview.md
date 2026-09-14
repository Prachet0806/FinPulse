# 01 — Overview

## What FinPulse is

FinPulse is an end-to-end analytics system that turns customer data into retention
strategies with quantified financial impact for fintech and banking organizations.

It bridges predictive modeling and business logic:

1. Understand the portfolio (segmentation).
2. Quantify risk and value per customer (churn + CLV + priority).
3. Recommend ROI-positive retention actions under budget/cap constraints.
4. Simulate gross and risk-adjusted impact.

## Who it is for

- Analytics product teams building decision-support tooling.
- Customer success / retention leaders allocating campaign budgets.
- Financial executives reviewing portfolio ROI and retention impact.

## Key capabilities

### Customer Intelligence

- **Behavioral Segmentation** (`intelligence/segmentation/`):
  K-Means with automated K-selection via silhouette score, median imputation,
  scaling, atomic model persistence + `metadata.json`.
- **Churn Prediction** (`intelligence/churn/`):
  RandomForest (`class_weight="balanced_subsample"`, `max_depth=12`,
  stratified split) with precision/recall/F1/ROC-AUC + confusion matrix,
  shared `FEATURE_COLS` for train/serve parity.
- **CLV Estimation** (`intelligence/clv/`):
  Retention-based `CLV = (ARPU * Retention) / (1 + Discount - Retention)`,
  fail-closed on missing churn, floored denominator, non-finite quarantine.

See: `07-intelligence-segmentation.md`, `08-intelligence-churn.md`,
`09-intelligence-clv-priority.md`.

### Decision Optimization — V2 Enterprise Engine

Four sequential layers (`decision_engine/`):

1. Base Valuation
2. Risk Adjustment
3. Eligibility Rules (fail-closed)
4. Portfolio Allocation (greedy ROI-density + budget + caps)

See: `10-decision-engine.md`.

### Business Impact

- **Revenue Impact Simulation** (`simulation/`):
  gross retention gain and risk-adjusted net (`gross − cost − risk`),
  quarantine of unknown strategies / NaN rows, never mutates input.
- **ROI semantics**: zero-cost + positive gain → `inf`;
  zero-cost + zero gain → `0.0`; NaN/negative guarded to `0.0`.
- **Strategy Breakdown**: counts, percents, unit costs.
- **Interactive Dashboard** (`dashboard/`): cached, schema-guarded,
  WebGL risk matrix, no remote assets.

See: `11-simulation.md`, `13-dashboard.md`.

## Dataset & domain adaptation

- **Source**: Telco Customer Churn dataset as proxy for subscription data.
- **Fintech mapping**: `MonthlyCharges → monthly_spend`,
  `tenure → account_tenure_months`, `Contract → account_plan`, etc.
- **Economic scaling**: 10x multiplier (`FINTECH_SCALE`) on
  `monthly_spend` / `lifetime_value` so CLVs land in the $2,000–$4,000 range
  and $100–$200 retention strategies can show positive ROI.
  Transform is idempotent.
- **Assumption**: contractual / semi-contractual relationships where churn
  is a discrete event (SaaS, banking, insurance-like).

## Business use cases

- **Banking & credit cards**: credit-limit increase vs. fee waiver trade-offs.
- **Insurance**: non-renewal risk + premium-discount cost/benefit.
- **B2B SaaS / subscriptions**: allocate CSM bandwidth by systemic impact.

## Limitations

- Static `effectiveness` / `cost` estimates — need A/B validation.
- First-order impact only — no network effects or macro shifts.
- Simplified universal discount rate for CLV.
- One-hot encoding currently happens before train/test split
  (inference reindexes to persisted training columns; full fit-on-train
  encoder is a tracked follow-up).

## Roadmap (V2)

- Next-best-offer engine (collaborative filtering + expected value).
- LP solver (PuLP/SciPy) replacing greedy allocator.
- MLOps migration from local `joblib` to MLflow registry.
