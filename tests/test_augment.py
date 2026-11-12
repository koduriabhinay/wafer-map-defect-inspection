import numpy as np

from src.augment import augment_batch, augment_pair
from src.data import generate_dataset


def _sample(n=16, seed=0):
    X, y = generate_dataset(n, seed=seed)
    return X, y


def test_augment_preserves_shape_and_binary():
    X, _ = _sample()
    rng = np.random.default_rng(0)
    Xa = augment_batch(X, rng)
    assert Xa.shape == X.shape
    assert set(np.unique(Xa)).issubset({0.0, 1.0})


def test_augment_deterministic_with_seed():
    X, _ = _sample()
    a = augment_batch(X, np.random.default_rng(7))
    b = augment_batch(X, np.random.default_rng(7))
    c = augment_batch(X, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_augment_pair_keeps_labels():
    X, y = _sample()
    Xa, ya = augment_pair(X, y, np.random.default_rng(0), p=1.0)
    assert np.array_equal(ya, y)
    assert Xa.shape == X.shape


def test_augment_actually_transforms_some_maps():
    # With p=1.0 and rotation/flip randomness, at least one map must change.
    X, _ = _sample(n=64)
    Xa, _ = augment_pair(X, np.zeros(len(X), dtype=int), np.random.default_rng(0), p=1.0)
    assert not np.array_equal(Xa, X)
