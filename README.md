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

### Full CLI example (all available arguments)

The following command shows how to invoke `main_run.py` with every supported argument.

```bash
python experiments/main_run.py \
  --mode train \
  --datasets ItalyPowerDemand,GunPoint,ECG5000,InlineSkate,ElectricDevices \
  --benchmarks "1NN-DTW,BOSS-ensemble,Rocket" \
  --data-dir data \
  --output-csv results/benchmark_comparison.csv \
  --model-dir trained_models \
  --predictions-dir results/predictions \
  --n-estimators 300 \
  --seed 7 \
  --n-train 100 \
  --n-test 40 \
  --n-timestamps 120
```

> **Note:** When running on multiple datasets, output CSVs are now named per dataset to avoid overwriting. You can also use `{dataset}` in `--output-csv` to explicitly control where each dataset’s results go.

---

## 🧠 `main_run.py` modes (what each one does)

`experiments/main_run.py` supports 4 execution modes: `benchmarks`, `synthetic`, `train`, and `forecast`. Here’s what each mode does and when to use it:

| Mode | What it does | Output |
|------|--------------|--------|
| `benchmarks` (default) | Train all classifiers from scratch and evaluate on test split. | `results/benchmark_comparison*.csv` |
| `train` | Train and save all classifiers to disk. | `trained_models/<dataset>/<classifier>.joblib` + per-dataset CSV |
| `forecast` / `predict` | Load saved models and run inference on test split. | `results/predictions/<dataset>/<classifier>.csv` |
| `synthetic` | Run a quick random-data sanity check (no UCR data). | (no files) |

### Common flags (quick reference)

| Flag | Applies to | What it does | Example |
|------|------------|-------------|---------|
| `--datasets` | all modes | Select dataset(s) under `data/` | `--datasets ItalyPowerDemand,GunPoint` |
| `--benchmarks` | `benchmarks`, `train`, `forecast` | Limit which benchmark classifiers run | `--benchmarks Rocket,BOSS-ensemble` |
| `--load-all` | `train` | Load existing models automatically (no prompt) | `--mode train --load-all` |
| `--no-tsf` | `benchmarks`, `train`, `forecast` | Skip the TSF (ours) classifier | `--no-tsf` |
| `--model-dir` | `train`, `forecast` | Custom location for saving/loading models | `--model-dir my_models` |
| `--predictions-dir` | `forecast` | Custom location for predictions output | `--predictions-dir my_preds` |

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

---

## 📊 Dashboard (new Dash version)

The project now includes a new interactive dashboard implemented in `utils/dashboard/generate_dashboard.py`.

### Run dashboard locally

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Run:

```bash
python utils/dashboard/generate_dashboard.py \
  --results results/benchmark_comparison.csv \
  --data-dir data \
  --hp-dir results \
  --host 127.0.0.1 --port 8050 --debug
```

3) Open browser:

`http://127.0.0.1:8050`

### What the dashboard shows

- **Metrics tab**: classifier accuracy bar + line trend for the selected dataset.
- **Timing tab**: train/predict/total times, one-point bubble across all datasets, stacked horizontal bars, dataset profile.
- **Hyperparameter tab**: hyperparameter optimization table, scatter (accuracy vs elapsed), with best points emphasized.

### Cleanup

- `utils/dashboard/generate_dashboard_old.py` removed.
- legacy static `visualization/*.html` files removed.

> Authors: Ane Miren Arregi, Iker Bereziartua, Eneko Zabaleta (ASTD Project 3)
