# Wafer Map Defect Inspection Classifier

An ML pipeline for semiconductor wafer map defect inspection: classification,
unsupervised pattern discovery and excursion-lot detection.

**Data.** The wafer maps are synthetic. They are generated 64×64 binary defect
masks whose 9 patterns follow the WM-811K class taxonomy (`center`, `donut`,
`edge_ring`, `edge_loc`, `loc`, `scratch`, `random`, `near_full`, `none`). They
are meant for developing and demonstrating the pipeline, not for conclusions
about real fab data. See [Using real data](#using-real-data) to plug in WM-811K.

## Components

1. **Supervised classification**: a small PyTorch CNN trained with
   class-weighted cross-entropy (imbalanced 9-class taxonomy), rotation/flip
   augmentation, and a stratified train/val/test split with fixed seeds.
2. **Unsupervised pattern discovery**: k-means clustering on CNN penultimate-layer
   embeddings of *unlabeled* maps, with silhouette analysis to select k.
3. **Excursion-lot detection**: per-lot binomial tests of defect rate vs the
   fab-wide baseline, with Benjamini–Hochberg FDR control at 1%.
4. **Experiment tracking**: every training run is logged (MLflow when available,
   JSON-lines fallback otherwise) with accuracy, macro-F1, and per-class F1.

## Results

Run: `python -m src.run_experiment --n 4500 --epochs 6 --seed 42`
(a reduced run for speed; see Reproducibility below).

| config | test accuracy | test macro-F1 |
|---|---|---|
| cnn_no_augment | 0.9756 | 0.9676 |
| cnn_augment | **0.9800** | **0.9737** |
| cnn_augment_lr5e4 | 0.9733 | 0.9667 |

Per-class test F1 (best config, `cnn_augment`):

| class | F1 | class | F1 | class | F1 |
|---|---|---|---|---|---|
| center | 0.9636 | edge_loc | 0.9418 | random | 1.0000 |
| donut | 1.0000 | loc | 0.9655 | near_full | 1.0000 |
| edge_ring | 1.0000 | scratch | 0.8939 | none | 0.9986 |

Unsupervised discovery: silhouette selected **k=3** on 900 unlabeled test maps
(cluster sizes 79 / 795 / 26, a coarse grouping: the dominant "none" class
forms the large cluster while defect patterns split into two smaller groups).
Excursion detection: 31 of 300 lots flagged at BH FDR 1%
(injected-excursion recall 1.000, false-positive rate 0.0037).

## Project structure

```
src/
  data.py        # synthetic wafer map generation + stratified splits + class weights
  augment.py     # rotation/flip augmentation (label-preserving)
  model.py       # PyTorch CNN (WaferCNN) + embedding extraction
  train.py       # training loop, class-weighted loss, MLflow/JSON logging
  evaluate.py    # accuracy, macro-F1, per-class F1
  clustering.py  # k-means on embeddings + silhouette k-selection
  excursions.py  # binomial tests + Benjamini-Hochberg FDR
  run_experiment.py  # end-to-end experiment driver
tests/           # deterministic pytest suite (data, augmentation, model,
                 #   clustering, excursions, metrics, full pipeline)
experiments/     # results.json + mlruns/ (generated, gitignored)
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.run_experiment --n 4500 --epochs 6 --seed 42
pytest tests/ -q
```

View MLflow runs (when mlflow is installed):

```bash
mlflow ui --backend-store-uri experiments/mlruns
```

## Methodology

- **Data.** 64×64 binary masks; class priors are imbalanced (`none` ≈ 40%,
  `donut`/`near_full` ≈ 3% each), roughly mirroring WM-811K. Circular wafer
  mask applied; 0.8% label noise. Deterministic via `numpy.random.Generator`
  seeds.
- **Model.** `WaferCNN`: 3 conv blocks (32/64/128 channels) + 2 dense layers
  (~2.2M params). Class-weighted `CrossEntropyLoss`; Adam; best-checkpoint
  selection on validation macro-F1.
- **Augmentation.** Random 90° rotations + horizontal/vertical flips (p=0.5 per
  sample); geometry-preserving, so labels stay valid.
- **Clustering.** 256-d penultimate-layer embeddings of held-out maps;
  k-means (k=2..12) with silhouette selection, `random_state` fixed.
- **Excursions.** One-sided binomial test per lot vs fab-wide baseline rate;
  BH step-up at FDR=0.01.

## Reproducibility

- All randomness is seeded (`--seed`, default 42). The pytest suite is fully
  deterministic.
- The default run uses 4,500 synthetic maps and 6 epochs so the full pipeline
  finishes in minutes on CPU. Scale `--n`/`--epochs` up for a longer run.

## Using real data

No real fab data is used or distributed. To train on WM-811K, replace
`src/data.py::generate_map` with a loader that keeps the same
`(X: (n,64,64) float32 in [0,1], y: (n,) int64)` contract and 9-class label
mapping. The rest of the pipeline does not depend on the data source.

## License

MIT, see [LICENSE](LICENSE).
