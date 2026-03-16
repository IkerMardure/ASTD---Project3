# ASTD Project 3

A small Python project for experimenting with time series classification using **aeon**.

## ✅ What’s included

- `classifiers/tsf_classifier.py`: A lightweight wrapper around `aeon.classification.interval_based.TimeSeriesForestClassifier`.
- `data/download_ucr_datasets.py`: Download selected UCR benchmark datasets.
- `experiments/main_run.py`: Entry-point for running experiments (synthetic and real data).
- `results/`: Intended for storing plots, tables, and analysis outputs.

---

## 🚀 Quick start

### 1) Install dependencies

From the repo root:

```bash
pip install -r requirements.txt
```

### 2) Run a quick experiment

```bash
python experiments/main_run.py
```

Optional parameters (example):

```bash
python experiments/main_run.py --n-train 100 --n-test 40 --n-timestamps 120 --n-estimators 300 --seed 7
```

---

## 📥 Download recommended UCR datasets

To download the following datasets into `data/`:

- ItalyPowerDemand
- GunPoint
- ECG5000
- InlineSkate
- ElectricDevices

Run:

```bash
python data/download_ucr_datasets.py
```

---

## 🧩 Where to extend

- Add experiment orchestration and evaluation logic in `experiments/main_run.py` and `experiments/validation.py`.
- Add new classifiers or baselines under `classifiers/`.
- Store result outputs (plots, metrics, tables) in `results/`.

---

## � Visualizing time series

Use the helper functions in `utils/visualize_TS.py` to plot time series from any UCR-style dataset in `data/`.

### Option A — Python API (recommended)

```python
from utils.visualize_TS import load_ucr_txt_dataset, generate_one_graph, generate_dataset_graph

X, y = load_ucr_txt_dataset("data/ItalyPowerDemand/ItalyPowerDemand_TRAIN.txt")

# Plot a single series
generate_one_graph(X[0], dataset_name="ItalyPowerDemand", index=0, label=y[0], save=True)

# Plot a small grid of the first 16 series
generate_dataset_graph(X, dataset_name="ItalyPowerDemand", labels=y, max_series=16, save=True)
```

### Option B — CLI (quick and scriptable)

```bash
# Grid plot (default) of the first 16 series
env PYTHONPATH=. python -m utils.visualize_TS \
  --input data/ItalyPowerDemand/ItalyPowerDemand_TRAIN.txt \
  --mode grid \
  --max-series 16

# Single-series plot (index 3)
env PYTHONPATH=. python -m utils.visualize_TS \
  --input data/ItalyPowerDemand/ItalyPowerDemand_TRAIN.txt \
  --mode one --index 3
```

The generated PNGs are saved into the `visualization/` folder by default (or any folder you pass via `--out-dir`).

---

## �📝 Notes

- `AeonTSFClassifier` expects time series inputs as NumPy arrays. It accepts 2D arrays (univariate) or 3D arrays (multivariate).
- If `aeon` is not installed, importing `AeonTSFClassifier` raises an informative `ImportError`.
