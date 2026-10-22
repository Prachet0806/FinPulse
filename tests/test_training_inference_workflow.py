import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from configs.settings import RANDOM_SEED, TEST_SIZE, FINTECH_SCALE
from ingestion.validator import validate_data
from pipeline.model_bundle import load_bundle, score_new_customers, train_and_evaluate
import pipeline.model_bundle as model_bundle


def _historical_rows(n=80):
    rng = np.random.default_rng(17)
    rows = []
    for i in range(n):
        churn = "Yes" if i % 2 else "No"
        tenure = int(4 + i % 35)
        monthly = float(35 + i % 20 + rng.uniform(0, 1))
        rows.append({
            "customerID": f"C{i}",
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": str(monthly * tenure),
            "Contract": ["Month-to-month", "One year", "Two year"][i % 3],
            "PaymentMethod": ["Card", "Bank"][i % 2],
            "InternetService": "DSL",
            "Churn": churn,
        })
    return pd.DataFrame(rows)


def test_unlabeled_inference_is_batch_independent():
    bundle, evaluation, _ = train_and_evaluate(_historical_rows(), publish=False)
    assert len(evaluation[0]) == bundle["metadata"]["n_holdout"]
    assert bundle["metadata"]["metrics"]["evaluation_source"] == "separate_validation_frame"

    new = _historical_rows().iloc[[1]].drop(columns="Churn").copy()
    new["customerID"] = "NEW"
    new["Contract"] = "Unseen plan"
    first, _, _ = score_new_customers(new, bundle)
    extra = _historical_rows().iloc[[2, 3]].drop(columns="Churn").copy()
    extra["customerID"] = ["OTHER1", "OTHER2"]
    batch, _, _ = score_new_customers(pd.concat([new, extra]), bundle)
    assert first[0]["churn_probability"].iloc[0] == pytest.approx(
        batch[0]["churn_probability"].loc[new.index[0]]
    )
    assert first[0]["recommended_strategy"].iloc[0] in {
        "no_action", "cashback_offer", "fee_waiver", "credit_limit_increase", "loan_offer"
    }


def test_validator_rejects_nonfinite_and_conflicting_ids():
    raw = _historical_rows(8)
    raw.loc[0, "MonthlyCharges"] = np.inf
    conflicting = raw.iloc[[1]].copy()
    conflicting["MonthlyCharges"] = 999
    clean, quarantine, report = validate_data(
        pd.concat([raw, conflicting], ignore_index=True),
        return_report=True, impute=False,
    )
    assert "C0" not in clean["customerID"].tolist()
    assert "C1" not in clean["customerID"].tolist()
    assert report["n_conflicting_ids"] == 2
    assert len(quarantine) == 3


def test_bundle_rejects_mixed_feature_schema(tmp_path):
    bundle, _, _ = train_and_evaluate(_historical_rows(), publish=False)
    bundle["feature_columns"] = ["wrong"]
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="schema"):
        load_bundle(path)


def test_preprocessing_fits_only_training_rows():
    raw = _historical_rows()
    train, holdout = train_test_split(
        raw, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=raw["Churn"]
    )
    bundle, evaluation, _ = train_and_evaluate(raw, publish=False)
    assert set(train["customerID"]).isdisjoint(set(evaluation[0]["customerID"]))
    assert set(evaluation[0]["customerID"]) == set(holdout["customerID"])
    expected = pd.to_numeric(train["TotalCharges"]).median() * FINTECH_SCALE
    assert bundle["transformer"].total_median == pytest.approx(expected)


def test_published_bundle_and_snapshot_share_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(model_bundle, "BUNDLE_PATH", tmp_path / "bundle.joblib")
    monkeypatch.setattr(model_bundle, "EVALUATION_PATH", tmp_path / "evaluation.parquet")
    bundle, evaluation, _ = train_and_evaluate(_historical_rows(), publish=True)
    loaded = load_bundle()
    saved = pd.read_parquet(model_bundle.EVALUATION_PATH)
    assert loaded["metadata"]["run_id"] == bundle["metadata"]["run_id"]
    assert (saved["model_run_id"] == loaded["metadata"]["run_id"]).all()
    unlabeled = _historical_rows().iloc[:2].drop(columns="Churn")
    scored, _, _ = score_new_customers(unlabeled, loaded)
    assert len(scored[0]) == len(unlabeled)
