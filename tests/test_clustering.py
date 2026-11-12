import numpy as np

from src.clustering import cluster_embeddings, select_k_by_silhouette


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.3, size=(40, 8))
    b = rng.normal(5, 0.3, size=(40, 8))
    c = rng.normal(-5, 0.3, size=(40, 8))
    return np.vstack([a, b, c])


def test_clustering_is_deterministic():
    Z = _blobs()
    l1, _ = cluster_embeddings(Z, k=3, seed=1)
    l2, _ = cluster_embeddings(Z, k=3, seed=1)
    assert np.array_equal(l1, l2)
    assert set(np.unique(l1)) == {0, 1, 2}


def test_silhouette_selects_k_in_range():
    Z = _blobs()
    best_k, scores = select_k_by_silhouette(Z, k_min=2, k_max=6, seed=1)
    assert 2 <= best_k <= 6
    assert set(scores.keys()) == {2, 3, 4, 5, 6}
    # three well-separated blobs -> k=3 should win
    assert best_k == 3


def test_silhouette_scores_finite():
    Z = _blobs()
    _, scores = select_k_by_silhouette(Z, k_min=2, k_max=4, seed=1)
    assert all(np.isfinite(v) for v in scores.values())
