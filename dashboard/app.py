# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from configs.settings import LOG_FORMAT, LOG_LEVEL
from pipeline.orchestrator import PipelineOrchestrator

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------- PIPELINE RUNNER ----------

@st.cache_data
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

# Custom CSS for an "Industry-Grade" aesthetic
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
    
    /* Custom Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #333842;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- UI STRUCTURING ----------

try:
    df, impact, roi, breakdown = run_pipeline()
    
    # Sidebar Filters
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/line-chart.png", width=64) # Placeholder logo
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

    # Apply Filters
    filtered_df = df.copy()
    if selected_segment != "All":
        filtered_df = filtered_df[filtered_df["segment_label"] == selected_segment]
    filtered_df = filtered_df[filtered_df["churn_probability"] >= min_risk]

    # Main Header
    st.markdown('<div class="main-header">💳 FinPulse Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Retention Strategy & Business Impact Simulator</div>', unsafe_allow_html=True)

    # ---------- EXECUTIVE METRICS ----------
    
    # Recalculate impact based on filters (Approximation for filtered view)
    # The actual impact dict is global, so we'll recalculate specifically for the filtered view
    eff_map = {"cashback_offer": 0.25, "fee_waiver": 0.20, "credit_limit_increase": 0.18, "loan_offer": 0.22, "no_action": 0.0}
    cost_map = {"cashback_offer": 200, "fee_waiver": 100, "credit_limit_increase": 0, "loan_offer": 50, "no_action": 0}
    
    f_effectiveness = filtered_df["recommended_strategy"].map(eff_map)
    f_costs = filtered_df["recommended_strategy"].map(cost_map)
    f_total_gain = (filtered_df["churn_probability"] * filtered_df["estimated_clv"] * f_effectiveness).sum()
    f_total_cost = f_costs.sum()
    f_net_benefit = f_total_gain - f_total_cost
    f_roi = f_net_benefit / f_total_cost if f_total_cost > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Retention Benefit", f"${f_net_benefit:,.0f}")
    col2.metric("Projected ROI", f"{f_roi:.2f}x")
    col3.metric("Total Strategy Cost", f"${f_total_cost:,.0f}")
    col4.metric("Customers Targeted", f"{len(filtered_df[filtered_df['recommended_strategy'] != 'no_action']):,}")

    st.markdown("---")

    # ---------- VISUALIZATIONS ----------
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown('<div class="section-header">Portfolio Segments</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-header">Strategy Allocation</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-header">Risk vs. Value Matrix</div>', unsafe_allow_html=True)
    
    # Scatter plot
    fig_scatter = px.scatter(
        filtered_df,
        x="churn_probability",
        y="estimated_clv",
        color="recommended_strategy",
        size="priority_score",
        hover_data=["customerID", "segment_label"],
        opacity=0.7,
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
    
    st.markdown('<div class="section-header">Intervention Targets</div>', unsafe_allow_html=True)
    
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

except Exception as e:
    st.error(f"Error running pipeline: {e}")
    logger.error(f"Pipeline failure: {e}", exc_info=True)
