# intelligence/segmentation/segment_labels.py

import logging

logger = logging.getLogger(__name__)

_BUSINESS_LABELS_K4 = {
    0: "High Value Loyal",
    1: "High Value At Risk",
    2: "Low Value At Risk",
    3: "New Customers",
}


def label_segments(df):
    """
    Map segment IDs to business-friendly names.

    Stable business names only when k==4; otherwise generic Segment_{i}
    plus a warning (KMeans IDs are arbitrary across fits).
    """
    df = df.copy()
    ids = sorted(df["segment_id"].dropna().unique().tolist())
    if len(ids) == 4 and set(ids) == {0, 1, 2, 3}:
        label_map = dict(_BUSINESS_LABELS_K4)
    else:
        logger.warning(
            "k=%d != 4; using generic segment names (IDs are unstable across fits)",
            len(ids),
        )
        label_map = {i: f"Segment_{i}" for i in ids}

    df["segment_label"] = df["segment_id"].map(label_map)

    return df
