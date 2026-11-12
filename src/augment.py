"""Rotation/flip augmentation for wafer maps (label-preserving)."""

from __future__ import annotations

import numpy as np


def augment_batch(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random 90-degree rotation and optional flips to each map.

    Rotations by multiples of 90 degrees and axis flips keep the wafer
    geometry valid, so class labels are preserved.
    """
    out = np.empty_like(X)
    for i, m in enumerate(X):
        k = rng.integers(0, 4)
        m = np.rot90(m, k)
        if rng.random() < 0.5:
            m = np.flip(m, axis=0)
        if rng.random() < 0.5:
            m = np.flip(m, axis=1)
        out[i] = m
    return out


def augment_pair(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, p: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return (augmented maps, unchanged labels)."""
    mask = rng.random(len(X)) < p
    Xa = X.copy()
    if mask.any():
        Xa[mask] = augment_batch(X[mask], rng)
    return Xa, y.copy()
