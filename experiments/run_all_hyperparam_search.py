from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from aeon.datasets import load_from_ts_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hyperparameter_search import (
    _infer_n_timepoints,
    default_grid_space,
    run_hyperparameter_search,
    save_search_results,
)


def _pick_compact(values: list):
    """Keep a small representative subset for compact grid runs."""
    if len(values) <= 2:
        return values
    mid = values[len(values) // 2]
    return [values[0], mid, values[-1]]


def build_compact_grid(X: np.ndarray) -> dict[str, list]:
    n_timepoints = _infer_n_timepoints(X)
    full = default_grid_space(n_timepoints=n_timepoints)
    compact = {
        "n_estimators": _pick_compact(full["n_estimators"]),
        "min_interval_length": _pick_compact(full["min_interval_length"]),
        "n_intervals": _pick_compact(full["n_intervals"]),
        "max_interval_length": _pick_compact(full["max_interval_length"]),
        "n_jobs": full["n_jobs"],
        "random_state": full["random_state"],
    }
    return compact


def main() -> None:
    datasets = [
        "ECG5000",
        "ElectricDevices",
        "GunPoint",
        "InlineSkate",
        "ItalyPowerDemand",
    ]
    methods = ["grid", "random", "optuna"]

    summary: list[dict[str, object]] = []

    for dataset_name in datasets:
        print(f"\n{'#' * 70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'#' * 70}")

        train_path = PROJECT_ROOT / "data" / dataset_name / f"{dataset_name}_TRAIN.ts"
        test_path = PROJECT_ROOT / "data" / dataset_name / f"{dataset_name}_TEST.ts"

        X_train, y_train = load_from_ts_file(train_path)
        X_test, y_test = load_from_ts_file(test_path)

        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        compact_grid = build_compact_grid(X)

        print(f"Loaded dataset with shape: {X.shape}")
        print(f"Compact grid size: {np.prod([len(v) for v in compact_grid.values()])}")

        for method in methods:
            print(f"\n{'=' * 60}")
            print(f"Running {method} on {dataset_name}")
            print(f"{'=' * 60}")

            results = run_hyperparameter_search(
                X=X,
                y=y,
                method=method,
                cv=3,
                metric="accuracy",
                random_state=42,
                n_trials=8,
                n_iter=12,
                param_grid=compact_grid,
                verbose=True,
            )

            print("\nBest hyperparameters:")
            print(results["best_params"])
            print(f"Best score: {results['best_score']:.4f}")

            save_path = PROJECT_ROOT / "results" / f"{dataset_name}_{method}_results.json"
            save_search_results(results, save_path)
            print(f"Saved results to: {save_path}")

            summary.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "best_score": results["best_score"],
                    "best_params": results["best_params"],
                    "elapsed_seconds": results.get("elapsed_seconds", None),
                }
            )

    summary_path = PROJECT_ROOT / "results" / "all_datasets_hyperparam_summary.json"
    save_search_results({"summary": summary}, summary_path)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
