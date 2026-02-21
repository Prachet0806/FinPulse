# scripts/run_decision_engine.py

from ingestion.loader import load_data
from ingestion.validator import validate_data
from processing.feature_engineering import build_features
from intelligence.segmentation.segmenter import segment_customers
from intelligence.segmentation.segment_labels import label_segments

from intelligence.churn.churn_model import train_churn_model
from intelligence.churn.churn_scoring import score_churn
from intelligence.clv.clv_model import estimate_clv
from intelligence.customer_priority import compute_priority

from decision_engine.apply_strategies import apply_strategies


def main():
    raw_path = "data/raw/Telco-Customer-Churn.csv"

    df = load_data(raw_path)
    df = validate_data(df)
    df = build_features(df)

    df = segment_customers(df)
    df = label_segments(df)

    model = train_churn_model(df)
    df = score_churn(model, df)

    df = estimate_clv(df)
    df = compute_priority(df)

    df = apply_strategies(df)

    print(df[[
        "segment_label",
        "churn_probability",
        "estimated_clv",
        "recommended_strategy"
    ]].head())


if __name__ == "__main__":
    main()