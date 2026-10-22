# 08 — Churn model and scoring

The full training workflow splits validated rows before fitting features. `train_churn_model` fits a RandomForest on the training cohort and evaluates against a separate transformed holdout. Metrics include accuracy, precision, recall, F1, ROC-AUC, Brier score, confusion matrix, and cohort sizes. The model bundle contains the fitted model and its feature schema.

`score_churn` takes feature names from the passed model first and reindexes numeric inputs to those names. The fitted `FeatureTransformer` establishes categorical meaning before this alignment. Feedback outputs such as `churn_probability`, CLV, priority, and recommended strategy are excluded from model inputs.

The bundled dataset is cross-sectional. A random holdout checks offline proxy behavior but cannot establish performance on later customers or whether predicted probabilities are calibrated in deployment. Define an observation date, prediction horizon, and later-cohort evaluation before operational use.
