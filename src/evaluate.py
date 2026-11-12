"""Classification metrics: accuracy, macro-F1, per-class F1."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 9) -> dict:
    """Return accuracy, macro-F1, and per-class F1 (fixed length n_classes)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class = f1_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in per_class],
    }
