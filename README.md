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

> **Note:** When running on multiple datasets, output CSVs are now named per dataset to avoid overwriting. You can also use `{dataset}` in `--output-csv` to explicitly control where each dataset’s results go.

---

## 🧠 `main_run.py` modes (what each one does)

`experiments/main_run.py` supports 4 execution modes: `benchmarks`, `synthetic`, `train`, and `forecast`. Here’s what each mode does and when to use it:

- **`benchmarks`** (default): Trains each classifier from scratch on the UCR train split and evaluates on the UCR test split.
  - Produces a results CSV (default `results/benchmark_comparison.csv`).
  - Does **not** keep trained models or per-instance predictions.

- **`synthetic`**: Runs a very small synthetic experiment (random data) to verify things work.
  - Useful for quick sanity checks without using UCR data.

- **`train`**: Trains classifiers on the UCR train split and **saves the trained models** to disk.
  - Saved models live in `trained_models/<dataset>/<classifier>.joblib`.
  - Produces per-dataset results CSVs of training metrics.

- **`forecast`**: Loads the saved models and runs inference on the UCR test split.
  - Requires models to be present under `trained_models/` (i.e., run `--mode train` first).
  - Saves per-instance predictions to `results/predictions/<dataset>/<classifier>.csv`.

### Example workflows

**Train a model once, then forecast later**
```bash
python experiments/main_run.py --mode train --datasets ItalyPowerDemand
python experiments/main_run.py --mode forecast --datasets ItalyPowerDemand
```

**Run a quick benchmark (no model persistence)**
```bash
python experiments/main_run.py
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

## 📊 Visualizing results (predictions + confusion)

The project includes a helper module `utils/visualize_predictions.py` to visualize classifier predictions and errors.

### Option A — Python API (recommended)

```python
from utils.visualize_predictions import plot_overlay_by_correctness, plot_confusion_matrix

# Overlay plot: correct series are fully visible, incorrect ones are semi-transparent
plot_overlay_by_correctness(
    dataset_name="ItalyPowerDemand",
    predictions_csv="results/predictions/ItalyPowerDemand/1NN-DTW.csv",
    save=True,
)

# Confusion matrix (counts)
plot_confusion_matrix(
    y_true=[...],
    y_pred=[...],
    dataset_name="ItalyPowerDemand",
    save=True,
)
```

### Option B — CLI (quick and scriptable)

```bash
# Overlay: per-class colors, transparent = incorrect
python -m utils.visualize_predictions \
  --mode overlay \
  --predictions results/predictions/ItalyPowerDemand/1NN-DTW.csv \
  --dataset-name ItalyPowerDemand

# Overlay for a specific class only (e.g., class "1")
python -m utils.visualize_predictions \
  --mode overlay \
  --predictions results/predictions/ItalyPowerDemand/1NN-DTW.csv \
  --dataset-name ItalyPowerDemand \
  --include-labels 1

# Confusion matrix
python -m utils.visualize_predictions \
  --mode confusion \
  --predictions results/predictions/ItalyPowerDemand/1NN-DTW.csv \
  --dataset-name ItalyPowerDemand
```

Generated images are saved to `visualization/` by default (or `--out-dir`).

---

## 🧩 Where to extend

- Add experiment orchestration and evaluation logic in `experiments/main_run.py` and `experiments/validation.py`.
- Add new classifiers or baselines under `classifiers/`.
- Store result outputs (plots, metrics, tables) in `results/`.

---

## 🧠 Notes

- `AeonTSFClassifier` expects time series inputs as NumPy arrays. It accepts 2D arrays (univariate) or 3D arrays (multivariate).
- If `aeon` is not installed, importing `AeonTSFClassifier` raises an informative `ImportError`.
