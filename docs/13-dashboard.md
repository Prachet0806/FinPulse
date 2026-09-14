# 13 — Dashboard

Source: `dashboard/app.py`

Streamlit + Plotly Express executive UI. Cached pipeline run, schema-guarded,
no remote assets, WebGL risk matrix with downsampling, ref-ID error handling.

Launch:

```bash
streamlit run dashboard/app.py
```

## Pipeline runner

```python
@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def run_pipeline():
    return PipelineOrchestrator().run_full_pipeline()
```

Runs at most once per hour (`max_entries=1`). Returns
`(df, impact, roi, breakdown)`.

## Schema guard

Required columns:

```python
{"segment_label", "churn_probability", "estimated_clv",
 "priority_score", "recommended_strategy", "customerID"}
```

Missing → `ValueError("Pipeline output missing columns: ...")`,
caught by the global handler below.

## Layout

- **Page config**: `FinPulse | AI Retention`, wide layout, expanded sidebar.
- **Styling**: single dark-theme `<style>` block (sole `unsafe_allow_html`
  exception); hides Streamlit menu/footer; styles metric cards.
- **Sidebar**: vendored logo (`dashboard/assets/line-chart.png`, text fallback),
  `Customer Segment` selectbox (`All` + unique `segment_label`),
  `Minimum Churn Probability (%)` slider → `min_risk`,
  status (`🟢 Online`) + record count.
- **Header**: `💳 FinPulse Intelligence` /
  `Automated Retention Strategy & Business Impact Simulator`.
- **Global metrics** (unfiltered, straight from pipeline): caption with
  `net_benefit`, `net_risk_adjusted`, global ROI; expander with full
  `breakdown` dataframe.
- **Filtered metrics**: recompute gain/cost/net/ROI on the filtered frame
  using shared `STRATEGIES` effectiveness/cost maps:
  `Net Retention Benefit`, `Projected ROI`, `Total Strategy Cost`,
  `Customers Targeted` (`recommended_strategy != no_action`).
- **Visualizations**:
  - `Portfolio Segments`: donut (`px.pie`, `hole=0.6`, Pastel).
  - `Strategy Allocation`: horizontal bar (`px.bar`, Set2).
  - `Risk vs. Value Matrix`: `px.scatter(x=churn_probability, y=estimated_clv,
    color=recommended_strategy, size=priority_score,
    hover=[customerID, segment_label], render_mode="webgl")`.
    Downsamples to 2,000 rows (seed 42) when filtered view exceeds 5,000.
- **Intervention Targets**: top 100 by `priority_score` desc with
  `customerID, segment_label, churn_probability (ProgressColumn),
  estimated_clv ($), priority_score, recommended_strategy`.

Filtering is a mask chain (`churn_probability >= min_risk` + optional segment),
no intermediate copies.

## Error handling

```python
except Exception:
    ref_id = uuid.uuid4().hex[:8]
    logger.exception("Pipeline failure (ref %s)", ref_id)
    st.error(f"Pipeline failed (ref ID {ref_id}). Please contact support.")
```

Never leaks tracebacks to the UI; correlate `ref_id` with logs.

## Performance notes

- Cache (`ttl=3600`) avoids re-running the full pipeline per interaction.
- WebGL + sampling keeps the scatter responsive on large portfolios.
- Masks/filters are vectorized pandas; breakdown caption uses precomputed
  pipeline totals.
