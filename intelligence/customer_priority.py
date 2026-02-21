# intelligence/customer_priority.py

def compute_priority(df):
    """
    Priority score = churn risk × CLV
    """

    df["priority_score"] = (
        df["churn_probability"] *
        df["estimated_clv"]
    )

    return df