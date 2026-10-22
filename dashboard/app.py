# dashboard/app.py

import logging
import hmac
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from configs.settings import setup_logging
from decision_engine.strategy_library import STRATEGIES
from pipeline.model_bundle import EVALUATION_PATH, load_bundle
from simulation.impact_simulator import simulate_impact
from simulation.roi_calculator import compute_roi
from simulation.strategy_breakdown import strategy_impact_breakdown

setup_logging()
logger = logging.getLogger(__name__)

# ---------- PIPELINE RUNNER ----------

def run_pipeline():
    bundle = load_bundle()
    if not EVALUATION_PATH.is_file():
        raise FileNotFoundError("Evaluation data missing. Run the training pipeline first.")
    df = pd.read_parquet(EVALUATION_PATH)
    if "model_run_id" not in df or not (df["model_run_id"] == bundle["metadata"]["run_id"]).all():
        raise ValueError("Evaluation data and model bundle versions differ")
    impact = simulate_impact(df)
    return df, impact, compute_roi(impact), strategy_impact_breakdown(df), bundle["metadata"]


# ---------- PAGE CONFIG & STYLING ----------

st.set_page_config(
    page_title="FinPulse | AI Retention",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

dashboard_token = os.getenv("FINPULSE_DASHBOARD_TOKEN")
if not dashboard_token:
    st.error("Dashboard access is disabled. Configure FINPULSE_DASHBOARD_TOKEN.")
    st.stop()
with st.sidebar:
    supplied_token = st.text_input("Access token", type="password")
if not hmac.compare_digest(supplied_token, dashboard_token):
    st.info("Enter the access token to view portfolio data.")
    st.stop()

# Custom CSS for an "Industry-Grade" aesthetic (sole unsafe_allow_html exception)
st.markdown("""
<style>
    /* Main background and fonts */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1E2127;
        border: 1px solid #333842;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    div[data-testid="metric-container"] > div {
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- UI STRUCTURING ----------

try:
    df, impact, roi, breakdown, metadata = run_pipeline()

    # Schema guard (fail-closed: abort UI gracefully if contract broken)
    required = {
        "segment_label", "churn_probability", "estimated_clv",
        "priority_score", "recommended_strategy", "customerID",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pipeline output missing columns: {sorted(missing)}")

    # Sidebar Filters
    with st.sidebar:
        # Local vendored logo (no remote fetch); falls back to text
        logo = ROOT_DIR / "dashboard" / "assets" / "line-chart.png"
        if logo.is_file():
            st.image(str(logo), width=64)
        else:
            st.markdown("## 💳 FinPulse")
        st.markdown("## FinPulse Settings")
        st.markdown("---")

        # Segment Filter
        segments = ["All"] + list(df["segment_label"].unique())
        selected_segment = st.selectbox("Customer Segment", segments)

        # Risk Filter
        min_risk = st.slider("Minimum Churn Probability (%)", 0, 100, 0) / 100.0

        st.markdown("---")
        st.markdown("**Platform Status:** Evaluation snapshot loaded")
        st.markdown(f"**Data Build:** `{len(df):,} records`")
        st.markdown(f"**Model Run:** `{metadata['run_id'][:12]}`")

    # Apply Filters (mask chain, no intermediate copies)
    mask = df["churn_probability"] >= min_risk
    if selected_segment != "All":
        mask = mask & (df["segment_label"] == selected_segment)
    filtered_df = df[mask]

    # Main Header (native Streamlit, no unsafe HTML)
    st.title("💳 FinPulse Intelligence")
    st.subheader("Retention Strategy Scenario Explorer")
    st.warning("Scenario estimates use unvalidated treatment effects and proxy data. They are not measured retention gains.")
    st.caption("Random holdout evaluation on a Telco proxy; prediction horizon and real-world treatment effects are not established.")

    # ---------- EXECUTIVE METRICS ----------

    # Global (unfiltered) metrics straight from the pipeline — now actually used
    st.caption(
        f"Global scenario gross net: ${impact.get('net_benefit', 0):,.0f} | "
        f"Scenario risk-adjusted net: ${impact.get('net_risk_adjusted', 0):,.0f} | "
        f"Scenario gross ROI: {roi if roi != float('inf') else 'inf'}"
    )
    with st.expander("Global strategy breakdown"):
        st.dataframe(breakdown, width="stretch")
    with st.expander("Scenario assumptions"):
        st.dataframe(
            pd.DataFrame.from_dict(STRATEGIES, orient="index").reset_index(names="strategy"),
            width="stretch",
            hide_index=True,
        )

    f_impact = simulate_impact(filtered_df)
    f_total_cost = f_impact["total_strategy_cost"]
    f_net_benefit = f_impact["net_risk_adjusted"]
    f_roi = (f_net_benefit / f_total_cost) if f_total_cost else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Scenario Risk-Adjusted Net", f"${f_net_benefit:,.0f}")
    col2.metric("Scenario Risk-Adjusted ROI", f"{f_roi:.2f}x" if f_roi is not None else "N/A")
    col3.metric("Total Strategy Cost", f"${f_total_cost:,.0f}")
    col4.metric("Customers Targeted", f"{len(filtered_df[filtered_df['recommended_strategy'] != 'no_action']):,}")

    st.markdown("---")

    # ---------- VISUALIZATIONS ----------

    c1, c2 = st.columns([1, 1])

    with c1:
        st.header("Portfolio Segments")
        seg_counts = filtered_df["segment_label"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig_donut = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=20, b=20, l=20, r=20)
        )
    st.plotly_chart(fig_donut, width="stretch")

    with c2:
        st.header("Strategy Allocation")
        strat_counts = filtered_df["recommended_strategy"].value_counts().reset_index()
        strat_counts.columns = ["Strategy", "Count"]

        fig_bar = px.bar(
            strat_counts,
            x="Count",
            y="Strategy",
            orientation='h',
            color="Strategy",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="#333842"),
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_bar, width="stretch")

    st.header("Risk vs. Value Matrix")

    # Scatter plot (WebGL + downsample for large filtered views)
    scatter_df = filtered_df
    if len(scatter_df) > 5000:
        scatter_df = scatter_df.sample(2000, random_state=42)
    fig_scatter = px.scatter(
        scatter_df,
        x="churn_probability",
        y="estimated_clv",
        color="recommended_strategy",
        size="priority_score",
        hover_data=["customerID", "segment_label"],
        opacity=0.7,
        render_mode="webgl",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        xaxis=dict(title="Churn Probability", tickformat=".0%", showgrid=True, gridcolor="#333842"),
        yaxis=dict(title="Estimated CLV ($)", showgrid=True, gridcolor="#333842"),
        legend_title="Strategy",
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_scatter, width="stretch")

    # ---------- DATA TABLE ----------

    st.header("Scenario Targets")

    display_df = filtered_df.sort_values("priority_score", ascending=False).head(100)[[
        "customerID", "segment_label", "churn_probability", "estimated_clv", "priority_score", "recommended_strategy"
    ]]

    st.dataframe(
        display_df,
        column_config={
            "customerID": "Customer ID",
            "segment_label": "Segment",
            "churn_probability": st.column_config.ProgressColumn(
                "Risk",
                help="Probability of churning",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "estimated_clv": st.column_config.NumberColumn(
                "Est. CLV",
                format="$%d",
            ),
            "priority_score": st.column_config.NumberColumn(
                "Priority Score",
                format="%.2f"
            ),
            "recommended_strategy": "Action"
        },
        width="stretch",
        hide_index=True
    )

except Exception:
    ref_id = uuid.uuid4().hex[:8]
    logger.exception("Pipeline failure (ref %s)", ref_id)
    st.error(f"Pipeline failed (ref ID {ref_id}). Please contact support.")
