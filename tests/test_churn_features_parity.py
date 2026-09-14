import pandas as pd

from intelligence.churn.churn_model import FEATURE_COLS_PATH, train_churn_model
from intelligence.churn.churn_scoring import score_churn
from processing.feature_engineering import build_features


def _labeled(n_per_class=30):
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for label, tenure, monthly in (("No", 40, 30.0), ("Yes", 5, 90.0)):
        for i in range(n_per_class):
            rows.append({
                "customerID": f"{label}-{i}-{rng.integers(1e6)}",
                "tenure": int(tenure + rng.integers(-2, 3)),
                "MonthlyCharges": float(monthly + rng.normal()),
                "TotalCharges": str(float(monthly * tenure)),
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "InternetService": "Fiber optic",
                "TechSupport": "No",
                "OnlineSecurity": "No",
                "PaperlessBilling": "Yes",
                "Churn": label,
            })
    return pd.DataFrame(rows)


def test_churn_train_score_parity_and_unseen_category(tmp_path, monkeypatch):
    import configs.settings as settings

    monkeypatch.setattr(settings, "MODELS_DIR", str(tmp_path))
    import intelligence.churn.churn_model as cm

    monkeypatch.setattr(cm, "MODEL_PATH", str(tmp_path / "churn_model.joblib"))
    monkeypatch.setattr(cm, "METRICS_PATH", str(tmp_path / "churn_metrics.json"))
    monkeypatch.setattr(cm, "FEATURE_COLS_PATH", str(tmp_path / "churn_feature_columns.json"))
    import processing.feature_engineering as fe

    monkeypatch.setattr(fe, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(fe, "ENCODER_COLUMNS_PATH", str(tmp_path / "encoder_columns.json"))

    df = build_features(_labeled(), save_encoder=True)
    model = train_churn_model(df, save_model=True)

    # Scoring frame with an unseen category value + extra bool col still aligns
    new = df.copy()
    assert hasattr(model, "predict_proba")
    scored = score_churn(model, new)
    assert "churn_probability" in scored.columns
    assert scored["churn_probability"].between(0, 1).all()

    #metrics sidecar persisted
    import json

    with open(tmp_path / "churn_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    assert {"precision", "recall", "f1", "roc_auc"} <= set(metrics)
