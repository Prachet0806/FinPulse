# intelligence/segmentation/segment_labels.py

import logging

logger = logging.getLogger(__name__)

def label_segments(df):
    """
    Map segment IDs to business-friendly names.

    KMeans IDs do not have stable business meaning. Keep labels neutral until
    a fitted cluster profile is reviewed and versioned with the model.
    """
    df = df.copy()
    ids = sorted(df["segment_id"].dropna().unique().tolist())
    label_map = {i: f"Segment_{i}" for i in ids}

    df["segment_label"] = df["segment_id"].map(label_map)

    return df
