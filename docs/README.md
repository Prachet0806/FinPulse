# FinPulse Documentation

Modular, function-by-function reference for the FinPulse Customer Decision Intelligence Platform.

## How to use these docs

- Start with [Overview](01-overview.md) if you are new.
- Follow [Getting Started](03-getting-started.md) to run the system.
- Jump to any functional module below for deep reference.

## Index

| # | Document | Function / Layer |
|---|----------|------------------|
| 01 | [Overview](01-overview.md) | Product purpose, capabilities, business use cases |
| 02 | [Architecture](02-architecture.md) | System layers, data flow, design principles |
| 03 | [Getting Started](03-getting-started.md) | Install, run pipeline, launch dashboard |
| 04 | [Configuration](04-configuration.md) | `configs/settings.py`, env vars, paths, constants |
| 05 | [Ingestion](05-ingestion.md) | `ingestion/` — loading + validation + quarantine |
| 06 | [Processing](06-processing.md) | `processing/` — fintech mapping + feature engineering |
| 07 | [Intelligence — Segmentation](07-intelligence-segmentation.md) | `intelligence/segmentation/` — clustering |
| 08 | [Intelligence — Churn](08-intelligence-churn.md) | `intelligence/churn/` — model + scoring |
| 09 | [Intelligence — CLV & Priority](09-intelligence-clv-priority.md) | `intelligence/clv/`, `customer_priority.py` |
| 10 | [Decision Engine](10-decision-engine.md) | `decision_engine/` — 4-layer optimization |
| 11 | [Simulation](11-simulation.md) | `simulation/` — impact, ROI, breakdown |
| 12 | [Pipeline](12-pipeline.md) | `pipeline/orchestrator.py` — end-to-end flow |
| 13 | [Dashboard](13-dashboard.md) | `dashboard/app.py` — Streamlit executive UI |
| 14 | [Scripts & CLI](14-scripts-cli.md) | `scripts/` entrypoints |
| 15 | [Testing](15-testing.md) | `tests/` coverage and conventions |

Related:

- Project README: `../README.md`
- Source root: `../`
- Raw data: `../data/raw/Telco-Customer-Churn.csv`
- Models: `../models/`
- Processed / quarantine output: `../data/processed/`
