
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifiers.tsf_classifier import AeonTSFClassifier, TSFConfig

# Optional dependency: Optuna
try:
    import optuna
except ImportError:
    optuna = None


# Metrics
def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "accuracy",
) -> float:
    """Compute evaluation metric."""
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    raise ValueError(f"Unsupported metric: {metric}")


# Model evaluation
def _infer_n_timepoints(X: np.ndarray) -> int:
    """Infer number of timepoints from 2D/3D time series arrays."""
    X_arr = np.asarray(X)
    if X_arr.ndim == 2:
        return int(X_arr.shape[1])
    if X_arr.ndim == 3:
        return int(X_arr.shape[2])
    raise ValueError(f"Expected a 2D or 3D array for X, got shape {X_arr.shape}")


def _build_tsf_config(
    params: dict[str, Any],
    n_timepoints: int,
) -> TSFConfig:
    """Create a TSFConfig while keeping interval limits valid for the series length."""
    min_interval_length = int(params["min_interval_length"])
    max_interval_raw = params.get("max_interval_length")
    max_interval_length: int | None

    if max_interval_raw is None:
        max_interval_length = None
    else:
        max_interval_length = int(max_interval_raw)
        max_interval_length = max(1, min(max_interval_length, n_timepoints))
        # Keep max interval at least as large as min interval.
        max_interval_length = max(max_interval_length, min_interval_length)

    return TSFConfig(
        n_estimators=int(params["n_estimators"]),
        min_interval_length=min_interval_length,
        n_intervals=params.get("n_intervals"),
        max_interval_length=max_interval_length,
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=int(params.get("random_state", 42)),
    )


def evaluate_single_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    metric: str = "accuracy",
) -> float:
    """Train TSF on one split and evaluate on validation data."""
    n_timepoints = _infer_n_timepoints(X_train)
    config = _build_tsf_config(params=params, n_timepoints=n_timepoints)

    clf = AeonTSFClassifier(config=config)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    return compute_metric(y_val, y_pred, metric=metric)


def cross_validate_params(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    cv: int = 5,
    metric: str = "accuracy",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Evaluate one hyperparameter configuration using Stratified K-Fold CV.

    ---
    For these hyperparameters, train and test the model multiple times 
    on different splits, then average the results to see how good they really are
    ---
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    fold_scores: list[float] = []

    for train_idx, val_idx in splitter.split(X_arr, y_arr):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        score = evaluate_single_split(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params=params,
            metric=metric,
        )
        fold_scores.append(score)

    return {
        "params": dict(params),
        "mean_score": float(np.mean(fold_scores)),
        "std_score": float(np.std(fold_scores)),
        "fold_scores": [float(s) for s in fold_scores],
    }


# Search spaces
def _default_n_intervals_candidates(n_timepoints: int) -> list[int]:
    """Return conservative, dataset-relative n_intervals candidates."""
    sqrt_tp = max(4, int(round(math.sqrt(n_timepoints))))
    candidates = {
        max(4, sqrt_tp // 2),
        max(4, sqrt_tp),
        max(4, int(round(1.5 * sqrt_tp))),
        max(4, 2 * sqrt_tp),
    }
    return sorted(candidates)


def _default_max_interval_candidates(n_timepoints: int) -> list[int | None]:
    """Return conservative, dataset-relative max interval candidates."""
    fractions = [0.25, 0.5, 0.75, 1.0]
    candidates = {
        max(1, int(round(n_timepoints * frac)))
        for frac in fractions
    }
    return [None] + sorted(candidates)


def default_grid_space(n_timepoints: int) -> dict[str, list[Any]]:
    """
    Discrete search space for grid/random search.
    """
    return {
        #these are the actual hyperparameters that we are tuning in our TSF model:
        "n_estimators": [50, 100, 200, 300, 500], #number of trees in the forest
        "min_interval_length": [3, 5, 7, 10, 15], #minimum size of time intervals extracted from the series
        "n_intervals": _default_n_intervals_candidates(n_timepoints), #number of random intervals sampled per tree
        "max_interval_length": _default_max_interval_candidates(n_timepoints), #upper bound on interval size (None keeps aeon default)
        "n_jobs": [-1], #This controls how many CPU cores the model uses --> n_jobs = -1: use all available cores
        "random_state": [42], #if you run the code twice, you should get the same splits and very similar same outcomes
    }


def default_optuna_space(
    trial: Any,
    n_timepoints: int,
) -> dict[str, Any]:
    """
    Search space for Optuna.
    """
    interval_candidates = _default_n_intervals_candidates(n_timepoints)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "min_interval_length": trial.suggest_int("min_interval_length", 3, 20),
        "n_intervals": trial.suggest_int(
            "n_intervals",
            min(interval_candidates),
            max(interval_candidates),
            step=1,
        ),
        "max_interval_length": trial.suggest_int(
            "max_interval_length",
            10,
            max(10, min(200, n_timepoints)),
            step=10,
        ),
        "n_jobs": -1,
        "random_state": 42,
    }


# Grid Search
def grid_search(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
    cv: int = 5,
    metric: str = "accuracy",
    random_state: int = 42,
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Exhaustive grid search over all parameter combinations.
    """
    if param_grid is None:
        n_timepoints = _infer_n_timepoints(X)
        param_grid = default_grid_space(n_timepoints=n_timepoints)

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    all_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    combinations = list(product(*values))
    total = len(combinations)

    if verbose:
        print(f"[Grid Search] Evaluating {total} combinations...")

    for idx, combo in enumerate(combinations, start=1):
        params = dict(zip(keys, combo))
        result = cross_validate_params(
            X=X,
            y=y,
            params=params,
            cv=cv,
            metric=metric,
            random_state=random_state,
        )
        all_results.append(result)

        if checkpoint_path is not None:
            _append_jsonl_row(
                {
                    "method": "grid_search",
                    "iteration": idx,
                    "params": params,
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"],
                },
                checkpoint_path,
            )

        if best_result is None or result["mean_score"] > best_result["mean_score"]:
            best_result = result

        if verbose:
            print(
                f"[{idx}/{total}] params={params} | "
                f"mean_{metric}={result['mean_score']:.4f} "
                f"+/- {result['std_score']:.4f}"
            )

    assert best_result is not None #Make sure best_result is not None

    return {
        "method": "grid_search",
        "metric": metric,
        "cv": cv,
        "best_params": best_result["params"],
        "best_score": best_result["mean_score"],
        "best_std": best_result["std_score"],
        "all_results": all_results,
    }


# Random Search
def sample_random_params(
    param_distributions: dict[str, list[Any]],
    rng: random.Random,
) -> dict[str, Any]:
    """Sample one random combination from a discrete parameter space."""
    return {
        key: rng.choice(values)
        for key, values in param_distributions.items()
    }


def random_search(
    X: np.ndarray,
    y: np.ndarray,
    param_distributions: dict[str, list[Any]] | None = None,
    n_iter: int = 20,
    cv: int = 5,
    metric: str = "accuracy",
    random_state: int = 42,
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Random search over a discrete hyperparameter space.
    """
    if param_distributions is None:
        n_timepoints = _infer_n_timepoints(X)
        param_distributions = default_grid_space(n_timepoints=n_timepoints)

    rng = random.Random(random_state)

    all_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    seen: set[tuple[tuple[str, Any], ...]] = set()

    #Total number of unique hyperparameter combinations possible
    max_unique = math.prod(len(v) for v in param_distributions.values())
    n_iter_effective = min(n_iter, max_unique)

    if verbose:
        print(
            f"[Random Search] Evaluating {n_iter_effective} random configurations "
            f"(max unique possible: {max_unique})..."
        )

    attempts = 0
    while len(all_results) < n_iter_effective and attempts < n_iter_effective * 10: #*10 = “try a bit harder to find unique configs, but don’t get stuck forever”
        attempts += 1
        params = sample_random_params(param_distributions, rng)
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)

        result = cross_validate_params(
            X=X,
            y=y,
            params=params,
            cv=cv,
            metric=metric,
            random_state=random_state,
        )
        all_results.append(result)

        if checkpoint_path is not None:
            _append_jsonl_row(
                {
                    "method": "random_search",
                    "iteration": len(all_results),
                    "params": params,
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"],
                },
                checkpoint_path,
            )

        if best_result is None or result["mean_score"] > best_result["mean_score"]:
            best_result = result

        if verbose:
            print(
                f"[{len(all_results)}/{n_iter_effective}] params={params} | "
                f"mean_{metric}={result['mean_score']:.4f} "
                f"+/- {result['std_score']:.4f}"
            )

    assert best_result is not None

    return {
        "method": "random_search",
        "metric": metric,
        "cv": cv,
        "n_iter": n_iter_effective,
        "best_params": best_result["params"],
        "best_score": best_result["mean_score"],
        "best_std": best_result["std_score"],
        "all_results": all_results,
    }


# Optuna Search
def optuna_search(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
    cv: int = 5,
    metric: str = "accuracy",
    random_state: int = 42,
    study_name: str = "tsf_optuna_search",
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Hyperparameter search with Optuna.
    """
    if optuna is None:
        raise ImportError(
            "Optuna is not installed. Install it with: pip install optuna"
        )

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n_timepoints = _infer_n_timepoints(X_arr)
    trial_results: list[dict[str, Any]] = []

    #1. Define the objective
    def objective(trial: Any) -> float:
        #2. It samples hyperparameters
        params = default_optuna_space(trial, n_timepoints=n_timepoints)
        #3. It evaluates them
        result = cross_validate_params(
            X=X_arr,
            y=y_arr,
            params=params,
            cv=cv,
            metric=metric,
            random_state=random_state,
        )

        #store extra evaluation details (std, folds) inside each Optuna trial
        trial.set_user_attr("std_score", result["std_score"])
        trial.set_user_attr("fold_scores", result["fold_scores"])

        #save all trial results in your own list for later use (analysis, plots, JSON)
        trial_results.append(result)

        if checkpoint_path is not None:
            _append_jsonl_row(
                {
                    "method": "optuna_search",
                    "trial": trial.number + 1,
                    "params": params,
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"],
                },
                checkpoint_path,
            )

        if verbose:
            print(
                f"[Trial {trial.number + 1}/{n_trials}] params={params} | "
                f"mean_{metric}={result['mean_score']:.4f} "
                f"+/- {result['std_score']:.4f}"
            )
        #4. It returns a score
        return result["mean_score"]

    #Defines the smart strategy Optuna uses to choose better hyperparameters
    sampler = optuna.samplers.TPESampler(seed=random_state)
    #Creates the optimization experiment that runs and tracks all trials (maximizing performance)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
    )
    #5. Optuna loop
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial

    return {
        "method": "optuna_search",
        "metric": metric,
        "cv": cv,
        "n_trials": n_trials,
        "best_params": dict(best_trial.params) | {"n_jobs": -1, "random_state": 42},
        "best_score": float(best_trial.value),
        "best_std": float(best_trial.user_attrs.get("std_score", 0.0)),
        "all_results": trial_results,
    }


# Utilities
def fit_best_model(
    X: np.ndarray,
    y: np.ndarray,
    best_params: dict[str, Any],
) -> AeonTSFClassifier:
    """Train final model on full dataset using best hyperparameters."""
    n_timepoints = _infer_n_timepoints(X)
    config = _build_tsf_config(params=best_params, n_timepoints=n_timepoints)

    clf = AeonTSFClassifier(config=config)
    clf.fit(X, y)
    return clf


def make_json_serializable(obj: Any) -> Any:
    """Convert numpy types into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _append_jsonl_row(row: dict[str, Any], path: str | Path) -> None:
    """Append one JSON object row to newline-delimited JSON file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as f:
        f.write(json.dumps(make_json_serializable(row), ensure_ascii=False))
        f.write("\n")


def save_search_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save search results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = make_json_serializable(results)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)


# Main public API
def run_hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "optuna",
    cv: int = 5,
    metric: str = "accuracy",
    random_state: int = 42,
    n_trials: int = 30,
    n_iter: int = 20,
    param_grid: dict[str, list[Any]] | None = None,
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Unified entry point for hyperparameter search.
    """
    start_time = time()

    if method == "optuna":
        results = optuna_search(
            X=X,
            y=y,
            n_trials=n_trials,
            cv=cv,
            metric=metric,
            random_state=random_state,
            verbose=verbose,
            checkpoint_path=checkpoint_path,
        )
    elif method == "grid":
        results = grid_search(
            X=X,
            y=y,
            param_grid=param_grid,
            cv=cv,
            metric=metric,
            random_state=random_state,
            verbose=verbose,
            checkpoint_path=checkpoint_path,
        )
    elif method == "random":
        results = random_search(
            X=X,
            y=y,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            metric=metric,
            random_state=random_state,
            verbose=verbose,
            checkpoint_path=checkpoint_path,
        )
    else:
        raise ValueError(
            "method must be one of {'optuna', 'grid', 'random'} "
            f"but got: {method}"
        )

    elapsed = time() - start_time
    results["elapsed_seconds"] = float(elapsed)

    return results


# Example usage with real datasets (UCR/UEA format)
if __name__ == "__main__":
    from aeon.datasets import load_from_ts_file
    datasets = [
        "ECG5000",
        "ElectricDevices",
        "GunPoint",
        "InlineSkate",
        "ItalyPowerDemand",
    ]
    methods = ["optuna", "grid", "random"]

    def _compact_values(values: list[Any]) -> list[Any]:
        if len(values) <= 2:
            return values
        return [values[0], values[-1]]

    summary_rows: list[dict[str, Any]] = []

    for dataset_name in datasets:
        print(f"\n{'#' * 70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'#' * 70}")

        # Load dataset (.ts format)
        train_path = PROJECT_ROOT / "data" / dataset_name / f"{dataset_name}_TRAIN.ts"
        test_path = PROJECT_ROOT / "data" / dataset_name / f"{dataset_name}_TEST.ts"

        X_train, y_train = load_from_ts_file(train_path)
        X_test, y_test = load_from_ts_file(test_path)

        # Convert to numpy (your classifier expects numpy)
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        print(f"Loaded dataset with shape: {X.shape}")

        # Keep grid search feasible while still exploring all tuned hyperparameters.
        n_timepoints = _infer_n_timepoints(X)
        base_grid = default_grid_space(n_timepoints=n_timepoints)
        compact_grid = {
            key: _compact_values(values)
            for key, values in base_grid.items()
        }
        total_combinations = math.prod(len(v) for v in compact_grid.values())
        print(f"Compact grid combinations: {total_combinations}")

        # Run hyperparameter search methods
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
                n_trials=12,
                n_iter=16,
                param_grid=compact_grid,
                verbose=True,
            )

            print("\nBest hyperparameters:")
            print(results["best_params"])
            print(f"Best score: {results['best_score']:.4f}")

            # Save results per dataset + method
            save_path = (
                PROJECT_ROOT
                / "results"
                / f"{dataset_name}_{method}_results.json"
            )

            save_search_results(results, save_path)

            print(f"Saved results to: {save_path}")

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "best_params": results["best_params"],
                    "best_score": results["best_score"],
                    "best_std": results.get("best_std", None),
                    "elapsed_seconds": results.get("elapsed_seconds", None),
                }
            )

    summary_path = PROJECT_ROOT / "results" / "tsf_hyperparameter_search_summary.json"
    save_search_results({"runs": summary_rows}, summary_path)
    print(f"Saved global summary to: {summary_path}")
