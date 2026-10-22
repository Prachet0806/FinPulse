# 06 — Feature processing

`processing/feature_engineering.py:FeatureTransformer` is the fitted path used by training and inference. It renames Telco proxy columns, applies the configured 10× scale, stores the training median for missing total charges, stores engagement normalization bounds, records categorical levels, and stores numeric medians. Its `transform` method reuses these values on holdout and inference rows.

All fitted categorical levels receive explicit dummy columns. Unknown inference categories produce zeros across the known levels; the reference category cannot change with batch composition. The `tenure_segment_new` column is retained for eligibility rules. Zero tenure creates `is_zero_tenure`, and its undefined monthly average is filled by the fitted numeric policy for model input.

`build_features` remains available for standalone demonstration scripts. It generates all dummy levels and can write a legacy encoder-column file. The supported full workflow uses `FeatureTransformer` in the versioned model bundle.
