# intelligence/segmentation/segment_labels.py

def label_segments(df):
    """
    Map segment IDs to business-friendly names.
    """

    label_map = {
        0: "High Value Loyal",
        1: "High Value At Risk",
        2: "Low Value At Risk",
        3: "New Customers"
    }

    df["segment_label"] = df["segment_id"].map(label_map)

    return df