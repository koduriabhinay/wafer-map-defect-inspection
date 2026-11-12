import numpy as np

from src.data import N_CLASSES
from src.evaluate import compute_metrics


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert all(v == 1.0 for v in m["per_class_f1"])


def test_per_class_f1_has_nine_entries():
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 2])
    m = compute_metrics(y_true, y_pred)
    assert len(m["per_class_f1"]) == N_CLASSES
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_metrics_keys_present():
    m = compute_metrics(np.array([0, 1]), np.array([1, 0]))
    assert set(m.keys()) == {"accuracy", "macro_f1", "per_class_f1"}
