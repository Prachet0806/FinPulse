# 07 — Segmentation

`intelligence/segmentation/segmenter.py` clusters account tenure, monthly spend, lifetime value, and engagement score. Training chooses K by silhouette search from 2 through 6, sampling at most 2,000 rows for each score, then fits a scaler and KMeans on training records. The bundle retains the scaler, KMeans model, and training medians.

`apply_segment_models` transforms later cohorts using those fitted values. It does not refit or use the scoring batch's medians. `label_segments` assigns neutral `Segment_i` names. Cluster IDs are arbitrary, so business names require separately reviewed, versioned profile rules.

`segment_customers(..., use_stored=True)` remains a legacy standalone path; the supported inference workflow loads the complete bundle.
