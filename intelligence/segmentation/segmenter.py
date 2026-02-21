# intelligence/segmentation/segmenter.py

import pandas as pd
import os
import joblib
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from configs.settings import RANDOM_SEED, DEFAULT_CLUSTERS, MODELS_DIR, LOG_FORMAT, LOG_LEVEL

from sklearn.metrics import silhouette_score

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def find_optimal_k(X_scaled, min_k=2, max_k=6):
    """
    Find optimal number of clusters using Silhouette Score.
    """
    logger.info(f"Analyzing optimal K between {min_k} and {max_k}...")
    
    best_k = min_k
    best_score = -1
    
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        logger.info(f"K={k}, Silhouette Score={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    logger.info(f"Optimal K identified: {best_k}")
    return best_k


def segment_customers(df: pd.DataFrame, save_models=True, auto_k=True) -> pd.DataFrame:
    """
    Segment customers using behavioral and value features.
    """

    logger.info("Starting customer segmentation...")

    features = [
        "account_tenure_months",
        "monthly_spend",
        "lifetime_value",
        "engagement_score"
    ]

    X = df[features].fillna(0)

    # Scale features for clustering
    scaler = StandardScaler()
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
        scaler_path = os.path.join(MODELS_DIR, "segment_scaler.joblib")
        kmeans_path = os.path.join(MODELS_DIR, "segment_kmeans.joblib")
        joblib.dump(scaler, scaler_path)
        joblib.dump(kmeans, kmeans_path)
        logger.info(f"Segmentation models saved to {MODELS_DIR}")

    logger.info(f"Segmentation complete. Created {n_clusters} clusters.")
    return df


def load_segmentation_models():
    """Load segmentation models from disk."""
    scaler_path = os.path.join(MODELS_DIR, "segment_scaler.joblib")
    kmeans_path = os.path.join(MODELS_DIR, "segment_kmeans.joblib")
    
    if os.path.exists(scaler_path) and os.path.exists(kmeans_path):
        return joblib.load(scaler_path), joblib.load(kmeans_path)
    return None, None