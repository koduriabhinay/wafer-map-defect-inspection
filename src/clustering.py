"""Unsupervised pattern discovery: k-means on model embeddings.

CNN/MLP hidden-layer embeddings of *unlabeled* wafer maps are clustered with
k-means; the silhouette score selects k. This surfaces defect patterns without
using labels, mirroring the "surface defect patterns for process
characterization" step of the project spec.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def extract_embeddings(model, X: np.ndarray) -> np.ndarray:
    """Hidden-layer embeddings for each map. Model-agnostic via duck typing."""
    return model.embed(X)


def select_k_by_silhouette(
    Z: np.ndarray, k_min: int = 2, k_max: int = 12, seed: int = 123
) -> tuple[int, dict[int, float]]:
    """Pick k maximizing the silhouette score over [k_min, k_max]."""
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(Z)
        # Silhouette needs at least 2 non-trivial clusters.
        if len(np.unique(labels)) < 2:
            scores[k] = float("-inf")
        else:
            scores[k] = float(silhouette_score(Z, labels))
    best_k = max(scores, key=lambda k: scores[k])
    return best_k, scores


def cluster_embeddings(
    Z: np.ndarray, k: int, seed: int = 123
) -> tuple[np.ndarray, KMeans]:
    """Deterministic k-means clustering of embeddings."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(Z), km
