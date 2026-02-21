# intelligence/segmentation/segment_insights.py

def generate_segment_insights(df):
    """
    Compute summary statistics per segment.
    """

    insights = df.groupby("segment_label").agg({
        "monthly_spend": "mean",
        "lifetime_value": "mean",
        "engagement_score": "mean",
        "account_tenure_months": "mean"
    }).reset_index()

    return insights