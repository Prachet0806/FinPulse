# scripts/run_segmentation.py

from ingestion.loader import load_data
from ingestion.validator import validate_data
from processing.feature_engineering import build_features
from intelligence.segmentation.segmenter import segment_customers
from intelligence.segmentation.segment_labels import label_segments
from intelligence.segmentation.segment_insights import generate_segment_insights


def main():
    raw_path = "data/raw/telco_churn.csv"

    df = load_data(raw_path)
    df = validate_data(df)
    df = build_features(df)

    df = segment_customers(df)
    df = label_segments(df)

    insights = generate_segment_insights(df)

    print(df[["segment_id", "segment_label"]].head())
    print("\nSegment Insights:")
    print(insights)


if __name__ == "__main__":
    main()