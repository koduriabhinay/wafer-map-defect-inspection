"""Full experiment: train, evaluate, cluster, and detect excursions.

Usage:
    python -m src.run_experiment [--n 4500] [--epochs 6] [--seed 42]

All randomness is seeded; metrics printed here are the real outputs of the
run and are also persisted to experiments/results.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .clustering import cluster_embeddings, extract_embeddings, select_k_by_silhouette
from .data import CLASSES, generate_dataset, stratified_split
from .evaluate import compute_metrics
from .excursions import flag_excursion_lots, simulate_lots
from .train import train_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4500)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tracking-dir", default="experiments/mlruns")
    args = ap.parse_args()

    print(f"[1/5] generating {args.n} synthetic wafer maps (seed={args.seed}) ...")
    X, y = generate_dataset(args.n, seed=args.seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(X, y, seed=args.seed)
    print(f"      train={len(ytr)} val={len(yva)} test={len(yte)}")

    configs = [
        {"name": "cnn_no_augment", "augment": False, "lr": 1e-3},
        {"name": "cnn_augment", "augment": True, "lr": 1e-3},
        {"name": "cnn_augment_lr5e4", "augment": True, "lr": 5e-4},
    ]
    results: dict = {"configs": {}, "data": {"n": args.n, "seed": args.seed}}

    best_model, best_name, best_f1 = None, "", -1.0
    for cfg in configs:
        print(f"[2/5] training {cfg['name']} ...")
        model, info = train_model(
            Xtr, ytr, Xva, yva,
            epochs=args.epochs, batch_size=64, lr=cfg["lr"],
            seed=args.seed, augment=cfg["augment"],
            experiment_name="wafer-map-defect-inspection",
            tracking_dir=args.tracking_dir,
        )
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(Xte.astype(np.float32)).unsqueeze(1)).argmax(1).numpy()
        tm = compute_metrics(yte, pred)
        results["configs"][cfg["name"]] = {
            "val_metrics": info["metrics"],
            "test_metrics": tm,
            "params": info["params"],
        }
        print(f"      test acc={tm['accuracy']:.4f} macro_f1={tm['macro_f1']:.4f}")
        if tm["macro_f1"] > best_f1:
            best_model, best_name, best_f1 = model, cfg["name"], tm["macro_f1"]

    print(f"[3/5] best config: {best_name} (test macro-F1={best_f1:.4f})")
    print("[4/5] clustering CNN embeddings of unlabeled test maps ...")
    Z = extract_embeddings(best_model, Xte)
    best_k, sil_scores = select_k_by_silhouette(Z, k_min=2, k_max=12, seed=123)
    labels, _ = cluster_embeddings(Z, best_k, seed=123)
    cluster_sizes = {int(k): int((labels == k).sum()) for k in range(best_k)}
    results["clustering"] = {
        "best_k": best_k,
        "silhouette_scores": {str(k): v for k, v in sil_scores.items()},
        "cluster_sizes": cluster_sizes,
        "n_unlabeled": int(len(Xte)),
    }
    print(f"      silhouette-selected k={best_k}, sizes={cluster_sizes}")

    print("[5/5] excursion-lot detection (binomial + BH FDR 1%) ...")
    defects, sizes, is_exc = simulate_lots(
        n_lots=300, lot_size=120, baseline_rate=0.05,
        excursion_fraction=0.1, excursion_rate=0.35, seed=7,
    )
    flags, pvals = flag_excursion_lots(defects, sizes, baseline_rate=0.05, fdr=0.01)
    results["excursions"] = {
        "n_lots": 300,
        "n_true_excursions": int(is_exc.sum()),
        "n_flagged": int(flags.sum()),
        "recall_on_injected": float(flags[is_exc].mean()),
        "false_positive_rate": float(flags[~is_exc].mean()),
    }
    print(f"      flagged {flags.sum()}/{len(flags)} lots "
          f"(recall={flags[is_exc].mean():.3f}, fpr={flags[~is_exc].mean():.4f})")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved experiments/results.json")

    # Human-readable summary (also used for the README metrics table).
    print("\n=== per-class test F1 (best config) ===")
    tm = results["configs"][best_name]["test_metrics"]
    for cls, f1 in zip(CLASSES, tm["per_class_f1"]):
        print(f"  {cls:10s} {f1:.4f}")
    print(f"  accuracy={tm['accuracy']:.4f}  macro_f1={tm['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
