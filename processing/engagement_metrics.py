# processing/engagement_metrics.py

def compute_engagement_metrics(df):
    """
    Create additional engagement indicators.
    """

    # Digital-first customers
    df["is_digital_user"] = df["digital_adoption"].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    # Premium support users
    df["has_priority_support"] = df["priority_support"].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    return df