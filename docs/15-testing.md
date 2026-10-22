# 15 — Testing

Run `pytest -q`. CI runs the suite on Python 3.11 and requires `pip-audit` to pass. The audit currently reports Pillow advisories: Streamlit 1.64 requires Pillow below 13, while the reported fixes require Pillow 13. This is an open dependency compatibility blocker, not a waived check.

`tests/test_training_inference_workflow.py` checks unlabeled inference, stable scoring across different batch compositions, non-finite and conflicting input quarantine, and rejection of a mixed model/feature bundle. `tests/test_allocator.py` includes a case where exact assignment beats greedy density selection. Other tests cover the validator, feature engineering, CLV, eligibility, simulator, ROI, and train/score feature parity.

The bundled Telco file has no observation timestamps or treatment outcomes. Tests establish code behavior, not temporal prediction quality, causal effectiveness, production security, or financial validity. Those require representative dated data and deployment-specific checks.
