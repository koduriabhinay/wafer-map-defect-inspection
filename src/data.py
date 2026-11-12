"""Synthetic wafer map data generation.

SYNTHETIC-BY-DESIGN: the full WM-811K dataset is multiple GBs and was not
downloaded here. These maps are programmatically generated 64x64 binary
defect masks whose 9 defect patterns mirror the WM-811K class taxonomy
(center, donut, edge-ring, edge-loc, loc, scratch, random, near-full, none).
They are useful for pipeline development and demonstration, not for claims
about real fab data.
"""

from __future__ import annotations

import numpy as np

MAP_SIZE = 64
N_CLASSES = 9

CLASSES = [
    "center",
    "donut",
    "edge_ring",
    "edge_loc",
    "loc",
    "scratch",
    "random",
    "near_full",
    "none",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Imbalanced class prior, roughly mirroring WM-811K ("none" dominates,
# "donut"/"near_full" are rare).
CLASS_PRIORS = np.array([0.06, 0.03, 0.08, 0.10, 0.10, 0.08, 0.12, 0.03, 0.40])


def wafer_mask(size: int = MAP_SIZE) -> np.ndarray:
    """Circular wafer mask: 1 inside the wafer, 0 outside."""
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - size / 2 + 0.5) ** 2 + (yy - size / 2 + 0.5) ** 2)
    return (dist <= size / 2 - 1).astype(np.float32)


_WAFER_MASK = wafer_mask()


def _add_noise(m: np.ndarray, rng: np.random.Generator, p: float = 0.01) -> np.ndarray:
    flip = rng.random(m.shape) < p
    return np.clip(m + flip.astype(np.float32), 0, 1)


def _blob(m: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : m.shape[0], 0 : m.shape[1]]
    return np.maximum(m, (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= r).astype(np.float32))


def _ring(m: np.ndarray, cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : m.shape[0], 0 : m.shape[1]]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.maximum(m, ((d >= r_in) & (d <= r_out)).astype(np.float32))


def _scratch_line(m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = m.shape[0]
    x0, y0 = rng.uniform(4, size - 4, 2)
    angle = rng.uniform(0, np.pi)
    length = rng.uniform(size * 0.5, size * 0.95)
    width = rng.uniform(1.0, 2.5)
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = np.cos(angle), np.sin(angle)
    t = (xx - x0) * dx + (yy - y0) * dy
    perp = np.abs((xx - x0) * dy - (yy - y0) * dx)
    line = (np.abs(t) <= length / 2) & (perp <= width)
    return np.maximum(m, line.astype(np.float32))


def generate_map(class_name: str, rng: np.random.Generator, size: int = MAP_SIZE) -> np.ndarray:
    """Generate one binary wafer defect map for the given class."""
    c = size / 2
    m = np.zeros((size, size), dtype=np.float32)

    if class_name == "center":
        m = _blob(m, c + rng.uniform(-3, 3), c + rng.uniform(-3, 3), rng.uniform(6, 11))
    elif class_name == "donut":
        r = rng.uniform(10, 14)
        m = _ring(m, c, c, r, r + rng.uniform(2.5, 4.5))
    elif class_name == "edge_ring":
        m = _ring(m, c, c, size / 2 - rng.uniform(5, 7), size / 2 - 1.5)
    elif class_name == "edge_loc":
        ang = rng.uniform(0, 2 * np.pi)
        m = _blob(
            m,
            c + (size / 2 - 5) * np.cos(ang),
            c + (size / 2 - 5) * np.sin(ang),
            rng.uniform(4, 7),
        )
    elif class_name == "loc":
        m = _blob(
            m,
            rng.uniform(size * 0.3, size * 0.7),
            rng.uniform(size * 0.3, size * 0.7),
            rng.uniform(4, 8),
        )
    elif class_name == "scratch":
        m = _scratch_line(m, rng)
    elif class_name == "random":
        m = (rng.random((size, size)) < rng.uniform(0.04, 0.10)).astype(np.float32)
    elif class_name == "near_full":
        m = (rng.random((size, size)) < rng.uniform(0.75, 0.92)).astype(np.float32)
    elif class_name == "none":
        m = (rng.random((size, size)) < rng.uniform(0.002, 0.012)).astype(np.float32)
    else:
        raise ValueError(f"unknown class: {class_name}")

    m = _add_noise(m, rng, p=0.008)
    # Defects only exist on the wafer itself.
    return (m * _WAFER_MASK).astype(np.float32)


def generate_dataset(
    n: int, seed: int = 42, priors: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n wafer maps. Returns (X, y) with X in [0,1]^{n,64,64}, y in [0,9)."""
    rng = np.random.default_rng(seed)
    priors = CLASS_PRIORS if priors is None else np.asarray(priors, dtype=float)
    priors = priors / priors.sum()
    class_names = rng.choice(CLASSES, size=n, p=priors)
    X = np.stack([generate_map(c, rng) for c in class_names])
    y = np.array([CLASS_TO_IDX[c] for c in class_names], dtype=np.int64)
    return X, y


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Stratified train/val/test split. Returns (Xtr,ytr),(Xva,yva),(Xte,yte)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)

    def _stratified_take(frac: float, pool: np.ndarray) -> np.ndarray:
        take: list[int] = []
        for k in range(N_CLASSES):
            cls_idx = pool[y[pool] == k]
            rng.shuffle(cls_idx)
            n_take = max(1, int(round(len(cls_idx) * frac))) if len(cls_idx) else 0
            take.extend(cls_idx[:n_take].tolist())
        return np.array(sorted(take), dtype=np.int64)

    test_idx = _stratified_take(test_size, idx)
    remaining = np.setdiff1d(idx, test_idx)
    val_idx = _stratified_take(val_size / (1 - test_size), remaining)
    train_idx = np.setdiff1d(remaining, val_idx)
    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
        (X[test_idx], y[test_idx]),
    )


def class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights (mean-normalized) for imbalanced training."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    counts = np.maximum(counts, 1.0)
    w = len(y) / (N_CLASSES * counts)
    return (w / w.mean()).astype(np.float32)
