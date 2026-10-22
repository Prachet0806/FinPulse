# 13 — Dashboard

`dashboard/app.py` is a Streamlit view of the **held-out evaluation snapshot**. It loads `data/processed/evaluation.parquet` and the local model bundle, verifies their run IDs match, and never trains on page load.

`FINPULSE_DASHBOARD_TOKEN` must be configured. The app checks a password-style sidebar input before reading customer-level data. This shared token is a local safeguard; deploy behind proper user authentication and authorization for real records.

The dashboard offers segment and minimum-risk filters. It applies the shared `simulate_impact` function to the filtered rows, so global and filtered economics use the same assumptions and quarantine rules. Gross net and risk-adjusted net are labeled separately. An expander exposes strategy cost, effectiveness, penalty, and capacity assumptions.

The charts and top-100 table show held-out proxy customers. Values are scenario estimates, not observed retention impact. The view identifies the model run and states that prediction horizon and treatment effects are not established.
