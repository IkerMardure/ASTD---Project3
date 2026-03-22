from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from aeon.datasets import load_from_ts_file

from hyperparameter_search import (
    run_hyperparameter_search,
    save_search_results,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    train_path = DATA_DIR / dataset_name / f"{dataset_name}_TRAIN.ts"
    test_path = DATA_DIR / dataset_name / f"{dataset_name}_TEST.ts"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Dataset files not found for {dataset_name}")

    X_train, y_train = load_from_ts_file(train_path)
    X_test, y_test = load_from_ts_file(test_path)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    return X, y


def run_search_task(
    dataset_name: str,
    method: str,
    cv: int,
    n_trials: int,
    n_iter: int,
    metric: str,
    random_state: int,
    force: bool,
) -> dict[str, Any]:
    out_file = RESULTS_DIR / f"{dataset_name}_{method}_results.json"
    if out_file.exists() and not force:
        return {
            "dataset": dataset_name,
            "method": method,
            "status": "skipped",
            "path": str(out_file),
            "reason": "already exists",
        }

    try:
        X, y = load_dataset(dataset_name)

        results = run_hyperparameter_search(
            X=X,
            y=y,
            method=method,
            cv=cv,
            metric=metric,
            random_state=random_state,
            n_trials=n_trials,
            n_iter=n_iter,
            verbose=True,
        )

        save_search_results(results, out_file)

        return {
            "dataset": dataset_name,
            "method": method,
            "status": "done",
            "path": str(out_file),
            "best_score": results.get("best_score"),
            "best_params": results.get("best_params"),
        }

    except Exception as exc:
        return {
            "dataset": dataset_name,
            "method": method,
            "status": "error",
            "error": repr(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel hyperparameter search orchestration")
    parser.add_argument("--datasets", nargs="+", default=["ECG5000", "ElectricDevices", "GunPoint", "InlineSkate", "ItalyPowerDemand"], help="Dataset names")
    parser.add_argument("--methods", nargs="+", default=["optuna", "grid", "random"], help="Search methods")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna n_trials")
    parser.add_argument("--n-iter", type=int, default=20, help="Random search n_iter")
    parser.add_argument("--metric", default="accuracy", help="Metric")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--jobs", type=int, default=max(1, cpu_count() - 1), help="Number of parallel worker processes")
    parser.add_argument("--force", action="store_true", help="Overwrite existing result files")
    parser.add_argument("--dry-run", action="store_true", help="Print tasks without executing")

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for dataset_name in args.datasets:
        for method in args.methods:
            tasks.append(
                (
                    dataset_name,
                    method,
                    args.cv,
                    args.n_trials,
                    args.n_iter,
                    args.metric,
                    args.random_state,
                    args.force,
                )
            )

    if args.dry_run:
        print("Dry run tasks:")
        for t in tasks:
            print(f"dataset={t[0]}, method={t[1]}, cv={t[2]}, n_trials={t[3]}, n_iter={t[4]}")
        return

    print(f"Running {len(tasks)} tasks with {args.jobs} workers...")

    with Pool(processes=args.jobs) as pool:
        results = pool.starmap(run_search_task, tasks)

    print("--- Summary ---")
    done = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"done: {len(done)}, skipped: {len(skipped)}, errors: {len(errors)}")

    if errors:
        print("Errors:")
        for err in errors:
            print(err)


if __name__ == "__main__":
    main()
