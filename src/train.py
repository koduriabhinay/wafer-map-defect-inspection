"""Training loop with class-weighted loss, augmentation, and MLflow logging.

Logs every run to MLflow (params + metrics: accuracy, macro-F1, per-class F1)
under experiments/mlruns. If MLflow is unavailable, falls back to a local
JSON experiment log so results are still recorded.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .augment import augment_pair
from .data import class_weights
from .evaluate import compute_metrics
from .model import make_model

try:
    import mlflow

    _MLFLOW_OK = True
except Exception:  # pragma: no cover
    _MLFLOW_OK = False


def _to_loader(X, y, batch_size, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)).unsqueeze(1),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def train_model(
    Xtr,
    ytr,
    Xva,
    yva,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    augment: bool = True,
    experiment_name: str = "wafer-map-defect-inspection",
    tracking_dir: str = "experiments/mlruns",
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = make_model(seed=seed)
    weights = torch.from_numpy(class_weights(ytr))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = _to_loader(Xtr, ytr, batch_size, shuffle=True, seed=seed)
    history: dict[str, list[float]] = {"train_loss": [], "val_acc": [], "val_macro_f1": []}
    best_state, best_f1 = None, -1.0

    aug_rng = np.random.default_rng(seed + 1)
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            if augment:
                xa, ya = xb.numpy()[:, 0], yb.numpy()
                xa_aug, _ = augment_pair(xa, ya, aug_rng, p=0.5)
                xb = torch.from_numpy(xa_aug).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            va_t = torch.from_numpy(Xva.astype(np.float32)).unsqueeze(1)
            pred = model(va_t).argmax(1).numpy()
        m = compute_metrics(yva, pred)
        history["val_acc"].append(m["accuracy"])
        history["val_macro_f1"].append(m["macro_f1"])
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        va_t = torch.from_numpy(Xva.astype(np.float32)).unsqueeze(1)
        final_pred = model(va_t).argmax(1).numpy()
    final_metrics = compute_metrics(yva, final_pred)

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "augment": augment,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
    }
    _log_run(experiment_name, tracking_dir, params, final_metrics, history)
    return model, {"metrics": final_metrics, "history": history, "params": params}


def _log_run(experiment_name, tracking_dir, params, metrics, history):
    os.makedirs(tracking_dir, exist_ok=True)
    flat = {
        **{f"param_{k}": v for k, v in params.items()},
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        **{f"f1_class_{i}": v for i, v in enumerate(metrics["per_class_f1"])},
    }
    if _MLFLOW_OK:
        mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    **{f"f1_class_{i}": v for i, v in enumerate(metrics["per_class_f1"])},
                }
            )
    else:  # fallback when mlflow is not installed: local JSON log
        path = os.path.join(tracking_dir, "experiment_log.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps({"ts": time.time(), **flat}) + "\n")
