"""End-to-end smoke test: data -> train (tiny) -> metrics -> excursion flags."""

import os

import numpy as np

from src.clustering import cluster_embeddings, extract_embeddings, select_k_by_silhouette
from src.data import generate_dataset, stratified_split
from src.evaluate import compute_metrics
from src.excursions import flag_excursion_lots, simulate_lots
from src.train import train_model


def test_mini_pipeline_runs_and_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep mlruns/json logs out of the repo
    X, y = generate_dataset(360, seed=99)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(X, y, seed=99)

    model, info = train_model(
        Xtr, ytr, Xva, yva,
        epochs=2, batch_size=64, seed=99, augment=True,
        tracking_dir="experiments/mlruns",
    )
    m = info["metrics"]
    assert 0.0 <= m["accuracy"] <= 1.0
    assert len(m["per_class_f1"]) == 9
    assert len(info["history"]["train_loss"]) == 2

    # embeddings + clustering on unlabeled test maps
    Z = extract_embeddings(model, Xte)
    assert Z.shape == (len(Xte), 256)
    best_k, _ = select_k_by_silhouette(Z, k_min=2, k_max=5, seed=1)
    labels, _ = cluster_embeddings(Z, best_k, seed=1)
    assert len(labels) == len(Xte)

    # excursion detection on simulated lots
    defects, sizes, _ = simulate_lots(50, 60, 0.05, seed=5)
    flags, pvals = flag_excursion_lots(defects, sizes, 0.05, fdr=0.01)
    assert flags.shape == (50,) and pvals.shape == (50,)

    # test-set metrics sanity
    import torch

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte.astype(np.float32)).unsqueeze(1)).argmax(1).numpy()
    tm = compute_metrics(yte, pred)
    assert tm["accuracy"] > 0.15  # well above chance (1/9)
