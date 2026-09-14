# 07 — Intelligence: Segmentation

Sources (`intelligence/segmentation/`):

- `segmenter.py` → `segment_customers`, `find_optimal_k`, `load_segmentation_models`
- `segment_labels.py` → `label_segments`
- `segment_insights.py` → `generate_segment_insights`

Role in pipeline: Step 3 — cluster the portfolio into actionable segments.

## Features

```python
SEGMENT_FEATURES = [
    "account_tenure_months",
    "monthly_spend",
    "lifetime_value",
    "engagement_score",
]
```

All four must exist or `segment_customers` raises `ValueError`.

## `find_optimal_k(X_scaled, min_k=2, max_k=6)`

- Silhouette-score search over `k = min_k..max_k`, guarded for small N:
  - `n < 2` → `ValueError`.
  - `max_k = min(max_k, n-1)`; if `max_k < min_k` → warn + return `min_k`.
- Skips single-cluster labelings (silhouette undefined).
- `KMeans(n_clusters=k, random_state=42, n_init=10)` per candidate.
- Returns best-K by silhouette score.

## `segment_customers(df, save_models=True, auto_k=True, use_stored=False, use_robust_scaler=False)`

Flow:

1. **Reuse path** (`use_stored=True`):
   loads scaler + KMeans via `load_segmentation_models()`.
   If found: median-impute + add `is_<col>_missing` flags, then
   `scaler.transform` + `kmeans.predict` → `segment_id`.
   If missing: warn + fall through to refit.
2. **Tiny-frame guard**: `len(df) < 2` → warn + single segment `0`.
3. **Fit path**:
   - Per-column median imputation + `is_<col>_missing` flags
     (median NaN → `0.0`; never `fillna(0)` blindly).
   - Scale with `StandardScaler` (default) or `RobustScaler`
     (`use_robust_scaler=True`, better for skewed money axis).
   - K via `find_optimal_k` (`auto_k=True`) else `DEFAULT_CLUSTERS` (4).
   - `KMeans(...).fit_predict` → `segment_id`.
4. **Atomic persistence** (`save_models=True`):
   - `models/segment_scaler.joblib`, `models/segment_kmeans.joblib`
     via temp-file + `os.replace` (`_atomic_dump`).
   - `models/segmentation_metadata.json` with `k`, `silhouette`,
     `sklearn_version`, `scaler` (`"robust"`/`"standard"`), `features`.

Orchestrator calls:

```python
segment_customers(df, save_models=not use_stored, use_stored=use_stored)
```

so `use_stored=True` reuses artifacts instead of refitting.

## `label_segments(df)`

Maps `segment_id` → business-friendly `segment_label`:

- Exactly `k==4` with IDs `{0,1,2,3}` → stable names:
  `0: High Value Loyal`, `1: High Value At Risk`,
  `2: Low Value At Risk`, `3: New Customers`.
- Otherwise → generic `Segment_<i>` + warning
  (KMeans IDs are arbitrary across fits).

## `generate_segment_insights(df)`

Aggregation helper used for reporting/inspection (means/counts per segment).
See source for exact metrics.

## Artifacts

- `models/segment_scaler.joblib`
- `models/segment_kmeans.joblib`
- `models/segmentation_metadata.json`

## Edge cases

| Case | Behavior |
|------|----------|
| Missing segment feature | `ValueError` |
| `n < 2` | Single segment `0` (fit path) / `ValueError` in `find_optimal_k` |
| `use_stored=True`, no artifacts | Warning + refit |
| `k != 4` | Generic `Segment_<i>` labels |
