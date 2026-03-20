"""Mini GridSearch runner (rapido) usando hyperparameter_search.py internamente."""
from pathlib import Path
import sys

# Añadir la raíz del proyecto al PYTHONPATH para importar el módulo experiments.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hyperparameter_search import run_hyperparameter_search, save_search_results
from aeon.datasets import load_from_ts_file
import numpy as np

datasets = ["GunPoint", "ItalyPowerDemand"]
methods = ["grid", "random", "optuna"]

base_param_grid = {
    "n_estimators": [10, 30],
    "min_interval_length": [3, 5],
    "n_jobs": [-1],
    "random_state": [42],
}

for name in datasets:
    root = Path("data") / name
    X_train, y_train = load_from_ts_file(root / f"{name}_TRAIN.ts")
    X_test, y_test = load_from_ts_file(root / f"{name}_TEST.ts")
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    for method in methods:
        print(f"\n===== Dataset={name} method={method} =====")

        # El optuna se mueve a n_trials, random a n_iter.
        params = {
            "X": X,
            "y": y,
            "method": method,
            "cv": 3,
            "metric": "accuracy",
            "random_state": 42,
            "n_trials": 5 if method == "optuna" else 0,
            "n_iter": 6 if method == "random" else 0,
            "param_grid": base_param_grid,
            "verbose": True,
        }

        if method == "optuna":
            # Optuna no usa n_iter; se deja n_trials.
            params["n_iter"] = 0
        if method == "grid":
            params["n_trials"] = 0
            params["n_iter"] = 0

        res = run_hyperparameter_search(**params)

        suffix = f"{name}_{method}_quick_results.json"
        out_path = Path("results") / suffix
        save_search_results(res, out_path)
        print("Guardado:", out_path)
        print("mejor:", res["best_score"], res["best_params"])
