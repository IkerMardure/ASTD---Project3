# ASTD Project 3

A small Python project for experimenting with time series classification using **aeon**.

## ✅ Project Structure

- `classifiers/tsf_classifier.py`: Wrapper around aeon's `TimeSeriesForestClassifier` for consistent project usage.
- `classifiers/benchmarks/*.py`: Benchmark classifier specs (1NN-DTW, 1NN-ED, BOSS, Rocket, Shapelet, Catch22).
- `data/`: UCR dataset folders (raw `.ts`, `.txt`, `.arff`).
- `experiments/main_run.py`: Main orchestrator for training/eval/prediction across datasets.
- `experiments/validation.py`: Utilities for dataset I/O, model fit/predict/metrics, and save/load.
- `results/`: Output folder for results tables and predictions.
- `utils/dashboard/generate_dashboard.py`: Dash app to inspect results interactively.

---

## 🚀 Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run default benchmark

```bash
python experiments/main_run.py
```

This runs the default benchmark mode on `ItalyPowerDemand` (as set in code) and writes result CSVs.

---

## 🧪 CLI usage (`experiments/main_run.py`)

`main_run.py` supports modes: `benchmarks`, `train`, `predict`, `forecast`, `synthetic`.

### Common options

- `--mode [benchmarks|train|predict|forecast|synthetic]` (default: `benchmarks`)
- `--datasets` (comma-separated dataset names under `data/`)
- `--benchmarks` (comma-separated subset of benchmark names)
- `--data-dir`, `--model-dir`, `--predictions-dir`, `--output-csv`
- `--load-all` (train mode: auto-load existing models instead of prompt)
- `--no-tsf` (skip TSF model)
- `--jobs` dataset-level parallelism (`1` = serial)
- `--tsf-config` path to per-dataset TSF config JSON (n_estimators/min_interval_length)

### Example: Full data run (train + save)

```bash
python experiments/main_run.py \
  --mode train \
  --datasets ItalyPowerDemand,GunPoint,ECG5000,InlineSkate,ElectricDevices \
  --benchmarks 1NN-DTW,1NN-ED,BOSS-ensemble,Rocket \
  --tsf-config experiments/TSF_best_params.json \
  --load-all \
  --model-dir trained_models \
  --output-csv results/benchmark_comparison.csv \
  --predictions-dir results/predictions \
  --seed 42 \
  --jobs 4
```

### Example: Predict with existing models

```bash
python experiments/main_run.py \
  --mode predict \
  --datasets ItalyPowerDemand,GunPoint,ECG5000,InlineSkate,ElectricDevices \
  --benchmarks 1NN-DTW,1NN-ED,BOSS-ensemble,Rocket \
  --model-dir trained_models \
  --predictions-dir results/predictions \
  --output-csv results/predictions_summary.csv \
  --jobs 4
```

---

## 📝 Output conventions

- In **train**/benchmark flows, results are written to CSV by dataset:
  - `results/benchmark_comparison_<dataset>.csv`
- `predictions` folder contains per-sample outputs:
  - `results/predictions/<dataset>/<classifier>.csv`
- `predictions_summary_<dataset>.csv` tracks benchmark metrics per dataset.

`fit_time_s` is reported as training duration. If you use `--load-all` and model exists, `fit_time_s` is `0` because loaded model is not re-trained.

---

## 📊 Dashboard support

`utils/dashboard/generate_dashboard.py` now accepts patterns and multiple CSV files:

```bash
python utils/dashboard/generate_dashboard.py \
  --results 'results/benchmark_comparison_*.csv' \
  --data-dir data \
  --hp-dir results \
  --predictions-dir results/predictions \
  --viz-dir visualization \
  --host 127.0.0.1 --port 8050 --debug
```

It scans matches, concatenates all results, and provides cross-dataset metrics/time plotting.

---

## 📌 TSF per-dataset config

`experiments/TSF_best_params.json` stores TSF parameters by dataset:

```json
{
  "ECG5000": {"n_estimators": 1, "min_interval_length": 1},
  "ElectricDevices": {"n_estimators": 300, "min_interval_length": 3},
  "GunPoint": {"n_estimators": 100, "min_interval_length": 3},
  "InlineSkate": {"n_estimators": 200, "min_interval_length": 3},
  "ItalyPowerDemand": {"n_estimators": 300, "min_interval_length": 5}
}
```

---

## 🧪 1NN-DTW / 1NN-ED parallelization

- `classifiers/benchmarks/one_nn_dtw.py` now uses `n_jobs=-1`.
- `classifiers/benchmarks/one_nn_ed.py` now uses `n_jobs=-1`.

This allows euclidean/dtw nearest-neighbor evaluation parallel in new versions of aeon.

---

## 🧩 Notes on data organization

- UCR datasets available in `data/<dataset>`
- `assets/dashboard.css` for optional dashboard theme
- `utils/visualize_*` for more plotting

---

## ⚡ Quick failsafe

If you hit `FileNotFoundError` for `results/benchmark_comparison.csv`, run with the updated glob/pattern as above since benchmark-by-dataset output is now the default.

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

### Entrenamiento completo (todos los datasets + todas las técnicas)

Con este comando, se entrenan todos los clasificadores de benchmark y TSF, y se guardan modelos y CSV de resultados para cada dataset.

```bash
python experiments/main_run.py \
  --mode train \
  --datasets ItalyPowerDemand,GunPoint,ECG5000,InlineSkate,ElectricDevices \
  --tsf-config experiments/TSF_best_params.json \
  --load-all \
  --model-dir trained_models \
  --output-csv results/benchmark_comparison.csv \
  --predictions-dir results/predictions \
  --seed 42 \
  --jobs 4
```

Notas:
- `--load-all` evita prompts ya existentes y carga modelos existentes cuando estén presentes.
- La opción `--tsf-config` aplica `n_estimators` y `min_interval_length` específicos por dataset.
- Para ejecutar de nuevo desde cero, borra `trained_models/` antes o utiliza un directorio nuevo.


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
