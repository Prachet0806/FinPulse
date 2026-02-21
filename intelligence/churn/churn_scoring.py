# intelligence/churn/churn_scoring.py

def score_churn(model, df):
    """
    Generate churn probabilities.
    """

    X = df.select_dtypes(include=["int64", "float64"])
    probs = model.predict_proba(X)[:, 1]

    df["churn_probability"] = probs
    return df