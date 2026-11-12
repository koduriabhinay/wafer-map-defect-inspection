import numpy as np

from src.data import (
    CLASSES,
    CLASS_TO_IDX,
    MAP_SIZE,
    N_CLASSES,
    class_weights,
    generate_dataset,
    generate_map,
    stratified_split,
    wafer_mask,
)


def test_map_shape_and_binary():
    rng = np.random.default_rng(0)
    for c in CLASSES:
        m = generate_map(c, rng)
        assert m.shape == (MAP_SIZE, MAP_SIZE)
        assert set(np.unique(m)).issubset({0.0, 1.0})


def test_wafer_mask_is_circular():
    mask = wafer_mask()
    assert mask.shape == (MAP_SIZE, MAP_SIZE)
    # corners are outside the wafer, center is inside
    assert mask[0, 0] == 0 and mask[MAP_SIZE - 1, MAP_SIZE - 1] == 0
    assert mask[MAP_SIZE // 2, MAP_SIZE // 2] == 1


def test_all_nine_classes_generated():
    X, y = generate_dataset(2000, seed=1)
    assert X.shape == (2000, MAP_SIZE, MAP_SIZE)
    assert set(np.unique(y)) == set(range(N_CLASSES))
    assert len(CLASS_TO_IDX) == 9


def test_generation_is_deterministic():
    X1, y1 = generate_dataset(200, seed=123)
    X2, y2 = generate_dataset(200, seed=123)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    X3, _ = generate_dataset(200, seed=124)
    assert not np.array_equal(X1, X3)


def test_none_class_dominates():
    _, y = generate_dataset(5000, seed=2)
    counts = np.bincount(y, minlength=N_CLASSES)
    assert counts[CLASS_TO_IDX["none"]] == counts.max()
    assert counts[CLASS_TO_IDX["none"]] > counts[CLASS_TO_IDX["donut"]]


def test_stratified_split_disjoint_and_covering():
    X, y = generate_dataset(900, seed=3)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(X, y, seed=3)
    assert len(ytr) + len(yva) + len(yte) == len(y)
    # disjoint index sets -> concatenated labels multiset equals original
    assert sorted(np.concatenate([ytr, yva, yte]).tolist()) == sorted(y.tolist())
    # every class represented in every split (stratified)
    for ys in (ytr, yva, yte):
        assert set(np.unique(ys)) == set(range(N_CLASSES))


def test_class_weights_upweight_minority():
    _, y = generate_dataset(3000, seed=4)
    w = class_weights(y)
    assert w.shape == (N_CLASSES,)
    assert (w > 0).all()
    assert w[CLASS_TO_IDX["donut"]] > w[CLASS_TO_IDX["none"]]
    assert abs(w.mean() - 1.0) < 1e-6
