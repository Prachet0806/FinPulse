# dashboard/app.py

import logging
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
from pipeline.orchestrator import PipelineOrchestrator
from simulation.roi_calculator import compute_roi

setup_logging()
logger = logging.getLogger(__name__)

# ---------- PIPELINE RUNNER ----------

@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def run_pipeline():
    orchestrator = PipelineOrchestrator()
    # Ensure models are saved/loaded properly
    return orchestrator.run_full_pipeline()


# ---------- PAGE CONFIG & STYLING ----------

st.set_page_config(
    page_title="FinPulse | AI Retention",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    df, impact, roi, breakdown = run_pipeline()

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
        st.markdown("**Platform Status:** 🟢 Online")
        st.markdown(f"**Data Build:** `{len(df):,} records`")

    # Apply Filters (mask chain, no intermediate copies)
    mask = df["churn_probability"] >= min_risk
    if selected_segment != "All":
        mask = mask & (df["segment_label"] == selected_segment)
    filtered_df = df[mask]

    # Main Header (native Streamlit, no unsafe HTML)
    st.title("💳 FinPulse Intelligence")
    st.subheader("Automated Retention Strategy & Business Impact Simulator")

    # ---------- EXECUTIVE METRICS ----------

    # Global (unfiltered) metrics straight from the pipeline — now actually used
    st.caption(
        f"Global net benefit: ${impact.get('net_benefit', 0):,.0f} | "
        f"Risk-adjusted: ${impact.get('net_risk_adjusted', 0):,.0f} | "
        f"Global ROI: {roi if roi != float('inf') else 'inf'}"
    )
    with st.expander("Global strategy breakdown"):
        st.dataframe(breakdown, use_container_width=True)

    # Recalculate impact based on filters using the shared strategy registry
    eff_map = {s: p["effectiveness"] for s, p in STRATEGIES.items()}
    cost_map = {s: p["cost"] for s, p in STRATEGIES.items()}

    f_effectiveness = filtered_df["recommended_strategy"].map(eff_map)
    f_costs = filtered_df["recommended_strategy"].map(cost_map)
    f_total_gain = (filtered_df["churn_probability"] * filtered_df["estimated_clv"] * f_effectiveness).sum()
    f_total_cost = f_costs.sum()
    f_net_benefit = f_total_gain - f_total_cost
    f_roi = compute_roi({
        "total_strategy_cost": float(f_total_cost),
        "total_retention_gain": float(f_total_gain),
        "net_benefit": float(f_net_benefit),
    })

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Retention Benefit", f"${f_net_benefit:,.0f}")
    col2.metric("Projected ROI", f"{f_roi:.2f}x" if f_roi != float("inf") else "inf")
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
        st.plotly_chart(fig_donut, use_container_width=True)

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
        st.plotly_chart(fig_bar, use_container_width=True)

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
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------- DATA TABLE ----------

    st.header("Intervention Targets")

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
        use_container_width=True,
        hide_index=True
    )

except Exception:
    ref_id = uuid.uuid4().hex[:8]
    logger.exception("Pipeline failure (ref %s)", ref_id)
    st.error(f"Pipeline failed (ref ID {ref_id}). Please contact support.")
