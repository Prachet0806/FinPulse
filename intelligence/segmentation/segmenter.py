# intelligence/segmentation/segmenter.py

import json
import logging
import os
import tempfile

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from configs.settings import DEFAULT_CLUSTERS, MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

SEGMENT_FEATURES = [
    "account_tenure_months",
    "monthly_spend",
    "lifetime_value",
    "engagement_score",
]


def find_optimal_k(X_scaled, min_k=2, max_k=6):
    """
    Find optimal number of clusters using Silhouette Score.
    Guards against N < max_k.
    """
    n = X_scaled.shape[0]
    if n < 2:
        raise ValueError(f"Not enough rows for clustering (n={n})")
    max_k = min(max_k, n - 1)
    if max_k < min_k:
        logger.warning("N=%d too small; falling back to k=%d", n, min_k)
        return min_k
    logger.info("Analyzing optimal K between %d and %d...", min_k, max_k)

    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        # Silhouette undefined for single-cluster labelings
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels)
        logger.info("K=%d, Silhouette Score=%.4f", k, score)

        if score > best_score:
            best_score = score
            best_k = k

    logger.info("Optimal K identified: %d", best_k)
    return best_k


def _atomic_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def segment_customers(
    df: pd.DataFrame,
    save_models: bool = True,
    auto_k: bool = True,
    use_stored: bool = False,
    use_robust_scaler: bool = False,
) -> pd.DataFrame:
    """
    Segment customers using behavioral and value features.

    - Median imputation + missing flags (not fillna(0))
    - Optional RobustScaler for skewed money axis
    - Atomic persist of scaler+kmeans + metadata.json
    - use_stored=True transforms/predicts with persisted models
    """
    df = df.copy()
    logger.info("Starting customer segmentation...")

    missing = [c for c in SEGMENT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Segmentation missing features: {missing}")

    if use_stored:
        scaler, kmeans = load_segmentation_models()
        if scaler is not None and kmeans is not None:
            X = df[SEGMENT_FEATURES].copy()
            for col in SEGMENT_FEATURES:
                flag = f"is_{col}_missing"
                df[flag] = X[col].isna().astype(int)
                X[col] = X[col].fillna(X[col].median())
            df["segment_id"] = kmeans.predict(scaler.transform(X))
            logger.info("Segmentation applied from stored models.")
            return df
        logger.warning("use_stored=True but no stored models; refitting.")

    if len(df) < 2:
        logger.warning("Too few rows (%d); assigning single segment", len(df))
        df["segment_id"] = 0
        return df

    X = df[SEGMENT_FEATURES].copy()
    for col in SEGMENT_FEATURES:
        df[f"is_{col}_missing"] = X[col].isna().astype(int)
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        X[col] = X[col].fillna(med)

    # Scale features for clustering
    scaler = RobustScaler() if use_robust_scaler else StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Automated K selection
    if auto_k:
        n_clusters = find_optimal_k(X_scaled)
    else:
        n_clusters = DEFAULT_CLUSTERS

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    df["segment_id"] = kmeans.fit_predict(X_scaled)

    if save_models:
        import sklearn

        scaler_path = os.path.join(MODELS_DIR, "segment_scaler.joblib")
        kmeans_path = os.path.join(MODELS_DIR, "segment_kmeans.joblib")
        _atomic_dump(scaler, scaler_path)
        _atomic_dump(kmeans, kmeans_path)
        try:
            sil = (
                float(silhouette_score(X_scaled, df["segment_id"]))
                if n_clusters > 1
                else 0.0
            )
        except ValueError:
            sil = 0.0
        meta = {
            "k": int(n_clusters),
            "silhouette": sil,
            "sklearn_version": sklearn.__version__,
            "scaler": "robust" if use_robust_scaler else "standard",
            "features": SEGMENT_FEATURES,
        }
        meta_path = os.path.join(MODELS_DIR, "segmentation_metadata.json")
        fd, tmp = tempfile.mkstemp(dir=MODELS_DIR, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp, meta_path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        logger.info("Segmentation models saved to %s", MODELS_DIR)

    logger.info("Segmentation complete. Created %d clusters.", n_clusters)
    return df


def load_segmentation_models():
    """Load segmentation models from disk."""
    scaler_path = os.path.join(MODELS_DIR, "segment_scaler.joblib")
    kmeans_path = os.path.join(MODELS_DIR, "segment_kmeans.joblib")

    if os.path.exists(scaler_path) and os.path.exists(kmeans_path):
        return joblib.load(scaler_path), joblib.load(kmeans_path)
    return None, None
