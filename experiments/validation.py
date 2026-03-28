"""Validation utilities for comparing TS classifiers on UCR datasets.

This module defines a default benchmark suite and a small evaluation pipeline
to compare classifiers with consistent metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import joblib
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from classifiers.benchmarks.suite import (
	DEFAULT_BENCHMARK_SPECS,
	instantiate_benchmark,
)
from classifiers.tsf_classifier import AeonTSFClassifier, TSFConfig


def load_tsf_best_params(path: str | Path) -> dict[str, dict[str, Any]]:
	"""Load best TSF hyperparameter config per dataset from JSON summary file."""
	file_path = Path(path)
	if not file_path.exists():
		return {}
	with file_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		return {}

	best_runs = data.get("best_runs")
	if not isinstance(best_runs, dict):
		return {}

	result: dict[str, dict[str, Any]] = {}
	for dataset, dataset_run in best_runs.items():
		params = dataset_run.get("best_params") if isinstance(dataset_run, dict) else None
		if isinstance(params, dict):
			result[str(dataset)] = {
				"n_estimators": int(params.get("n_estimators", 200)),
				"min_interval_length": int(params.get("min_interval_length", 3)),
				"n_intervals": int(params.get("n_intervals", 10)),
				"max_interval_length": None if params.get("max_interval_length") is None else int(params.get("max_interval_length")),
				"n_jobs": int(params.get("n_jobs", -1)),
				"random_state": int(params.get("random_state", 42)),
			}
	return result


def _build_tsf_config_from_params(params: dict[str, Any], default_random_state: int | None = 42) -> TSFConfig:
	"""Build TSFConfig from params mapping with defaults and type conversion."""
	if params is None:
		params = {}

	n_estimators = int(params.get("n_estimators", 200))
	min_interval_length = int(params.get("min_interval_length", 3))
	n_intervals = int(params.get("n_intervals", 10))
	max_interval = params.get("max_interval_length")
	if max_interval is None:
		max_interval_length = None
	else:
		max_interval_length = int(max_interval)

	n_jobs = int(params.get("n_jobs", -1))
	random_state = int(params.get("random_state", default_random_state or 42))

	return TSFConfig(
		n_estimators=n_estimators,
		min_interval_length=min_interval_length,
		n_intervals=n_intervals,
		max_interval_length=max_interval_length,
		n_jobs=n_jobs,
		random_state=random_state,
	)


def _to_3d_numpy(X: np.ndarray) -> np.ndarray:
	"""Convert to aeon expected shape: (n_cases, n_channels, n_timepoints)."""
	arr = np.asarray(X)
	if arr.ndim == 2:
		return arr[:, np.newaxis, :]
	if arr.ndim == 3:
		return arr
	raise ValueError(f"X must be 2D or 3D. Received shape {arr.shape}.")


def _sanitize_filename(name: str) -> str:
	"""Create a filesystem-safe filename from a classifier/dataset name."""
	# Keep alphanumeric, dash, underscore, dot
	return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def _dataset_checkpoint_path(checkpoint_dir: str | Path | None, dataset_name: str) -> Path | None:
	"""Compute a per-dataset checkpoint path for incremental classifier results."""
	if checkpoint_dir is None:
		return None
	return Path(checkpoint_dir) / f"{_sanitize_filename(dataset_name)}_checkpoint.jsonl"


def _append_jsonl_row(row: dict[str, Any], path: Path) -> None:
	"""Append one JSON object row to newline-delimited JSON file."""
	_ensure_dir(path.parent)
	with path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(row, ensure_ascii=False))
		f.write("\n")


def _compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
	"""Compute classification metrics used in benchmark result rows."""
	y_true_arr = np.asarray(y_true)
	y_pred_arr = np.asarray(y_pred)

	return {
		"accuracy": float(np.mean(y_pred_arr == y_true_arr)),
		"precision_weighted": float(
			precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
		),
		"recall_weighted": float(
			recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
		),
		"f1_weighted": float(
			f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
		),
		"balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
	}


def _error_result_row(dataset_name: str, classifier_name: str, exc: Exception) -> dict[str, Any]:
	"""Build a standardized error row for failed evaluations."""
	return {
		"dataset": dataset_name,
		"classifier": classifier_name,
		"accuracy": np.nan,
		"precision_weighted": np.nan,
		"recall_weighted": np.nan,
		"f1_weighted": np.nan,
		"balanced_accuracy": np.nan,
		"fit_time_s": np.nan,
		"predict_time_s": np.nan,
		"status": "error",
		"error": f"{type(exc).__name__}: {exc}",
	}


def _prompt_load_or_retrain(model_path: Path, allow_load_all: bool = True) -> str:
	"""Ask the user whether to load an existing model or retrain (without saving).

	Returns:
	- "load": load this existing model
	- "retrain": retrain without saving/overwriting
	- "load_all": load this and all subsequent existing models without prompting
	"""
	prompt = (
		f"\nModel already exists at: {model_path}\n"
		"  [L]oad existing model (recommended)\n"
		"  [R]etrain (do not overwrite/save)\n"
	)
	if allow_load_all:
		prompt += "  [A]ll (load this and all remaining existing models)\n"
	prompt += "Choose [L/r" + ("/a" if allow_load_all else "") + "]: "

	while True:
		try:
			choice = input(prompt).strip().lower()
		except (EOFError, KeyboardInterrupt):
			# Non-interactive environment (e.g. CI), default to loading existing model.
			return "load"
		if choice == "" or choice.startswith("l"):
			return "load"
		if allow_load_all and choice.startswith("a"):
			return "load_all"
		if choice.startswith("r"):
			return "retrain"
		print("Please enter 'l' to load, 'r' to retrain" + (" or 'a' to load all" if allow_load_all else "") + ".")


def _model_path(model_dir: Path, dataset_name: str, classifier_name: str) -> Path:
	"""Return a full path where a trained model should be saved/loaded."""
	return Path(model_dir) / dataset_name / f"{_sanitize_filename(classifier_name)}.joblib"


def _predictions_path(predictions_dir: Path, dataset_name: str, classifier_name: str) -> Path:
	"""Return a full path where per-instance predictions should be saved."""
	return Path(predictions_dir) / dataset_name / f"{_sanitize_filename(classifier_name)}.csv"


def save_model(model: Any, path: Path) -> Path:
	"""Persist a trained model to disk."""
	_ensure_dir(path.parent)
	joblib.dump(model, path)
	return path


def load_model(path: Path) -> Any:
	"""Load a trained model previously saved with :func:`save_model`."""
	return joblib.load(path)


def save_predictions(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	output_path: Path,
	class_labels: list[str] | None = None,
) -> Path:
	"""Save per-instance predictions to a CSV file."""
	_ensure_dir(output_path.parent)

	# Ensure arrays are 1D and same length
	y_true = np.asarray(y_true).reshape(-1)
	y_pred = np.asarray(y_pred).reshape(-1)
	if y_true.shape[0] != y_pred.shape[0]:
		raise ValueError("y_true and y_pred must have the same length")

	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		headers = ["index", "y_true", "y_pred"]
		if class_labels is not None:
			headers.append("class_labels")
		writer.writerow(headers)
		for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
			row = [i, yt, yp]
			if class_labels is not None:
				row.append("|".join(class_labels))
			writer.writerow(row)

	return output_path


def load_ucr_txt_split(data_dir: str | Path, dataset_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
	"""Load a UCR split from a local .txt file.

	Expected path format:
	`<data_dir>/<dataset_name>/<dataset_name>_<split>.txt`
	where split is TRAIN or TEST.
	"""
	split_upper = split.upper()
	file_path = Path(data_dir) / dataset_name / f"{dataset_name}_{split_upper}.txt"
	if not file_path.exists():
		raise FileNotFoundError(
			f"Dataset split file not found: {file_path}. "
			"Make sure datasets are downloaded into data/."
		)

	arr = np.loadtxt(file_path)
	if arr.ndim == 1:
		arr = arr.reshape(1, -1)

	y = arr[:, 0]
	X = arr[:, 1:]
	return X, y


def load_ucr_dataset(data_dir: str | Path, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Load TRAIN and TEST splits for a UCR dataset from local files."""
	X_train, y_train = load_ucr_txt_split(data_dir=data_dir, dataset_name=dataset_name, split="TRAIN")
	X_test, y_test = load_ucr_txt_split(data_dir=data_dir, dataset_name=dataset_name, split="TEST")
	return X_train, y_train, X_test, y_test


def _evaluate_model(
	model: Any,
	model_name: str,
	dataset_name: str,
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	model_path: Path | None = None,
	predictions_path: Path | None = None,
) -> dict[str, Any]:
	"""Fit, predict and score one model, returning metrics.

	If `model_path` is provided, the fitted model is saved after training.
	If `predictions_path` is provided, per-instance predictions are also saved.
	"""
	X_train_3d = _to_3d_numpy(X_train)
	X_test_3d = _to_3d_numpy(X_test)

	print(f"  [{dataset_name}] {model_name} — fitting on {X_train_3d.shape[0]} samples...", flush=True)
	start_fit = time.perf_counter()
	model.fit(X_train_3d, y_train)
	fit_time_s = time.perf_counter() - start_fit

	if model_path is not None:
		save_model(model, model_path)
		print(f"  [{dataset_name}] {model_name} — saved model to {model_path}", flush=True)

	print(f"  [{dataset_name}] {model_name} — fit done ({fit_time_s:.1f}s). Predicting...", flush=True)

	start_pred = time.perf_counter()
	y_pred = model.predict(X_test_3d)
	predict_time_s = time.perf_counter() - start_pred

	if predictions_path is not None:
		save_predictions(y_test, y_pred, predictions_path)
		print(f"  [{dataset_name}] {model_name} — saved predictions to {predictions_path}", flush=True)

	metrics = _compute_classification_metrics(y_test, y_pred)
	accuracy = metrics["accuracy"]
	print(f"  [{dataset_name}] {model_name} — accuracy={accuracy:.4f}, predict={predict_time_s:.1f}s", flush=True)

	row: dict[str, Any] = {
		"dataset": dataset_name,
		"classifier": model_name,
		**metrics,
		"fit_time_s": fit_time_s,
		"predict_time_s": predict_time_s,
		"status": "ok",
		"error": "",
	}
	if model_path is not None:
		row["model_path"] = str(model_path)
	if predictions_path is not None:
		row["predictions_path"] = str(predictions_path)

	return row


def _evaluate_loaded_model(
	model: Any,
	model_name: str,
	dataset_name: str,
	X_test: np.ndarray,
	y_test: np.ndarray,
	predictions_path: Path | None = None,
) -> dict[str, Any]:
	"""Predict and score a pre-trained model, optionally saving predictions."""
	X_test_3d = _to_3d_numpy(X_test)

	print(f"  [{dataset_name}] {model_name} — predicting on {X_test_3d.shape[0]} samples...", flush=True)
	start_pred = time.perf_counter()
	y_pred = model.predict(X_test_3d)
	predict_time_s = time.perf_counter() - start_pred

	if predictions_path is not None:
		save_predictions(y_test, y_pred, predictions_path)
		print(f"  [{dataset_name}] {model_name} — saved predictions to {predictions_path}", flush=True)

	metrics = _compute_classification_metrics(y_test, y_pred)
	accuracy = metrics["accuracy"]
	print(f"  [{dataset_name}] {model_name} — accuracy={accuracy:.4f}, predict={predict_time_s:.1f}s", flush=True)

	row: dict[str, Any] = {
		"dataset": dataset_name,
		"classifier": model_name,
		**metrics,
		"fit_time_s": 0.0,
		"predict_time_s": predict_time_s,
		"status": "ok",
		"error": "",
	}
	if predictions_path is not None:
		row["predictions_path"] = str(predictions_path)
	return row


def run_benchmark_suite(
	dataset_name: str,
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
	ts_config_path: str | Path | None = "results/best_of_best_tsf.json",
	checkpoint_path: str | Path | None = None,
	model_dir: str | Path | None = None,
	predictions_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
	"""Run the default benchmark suite on one dataset.

	If a model cannot be instantiated or trained due to missing optional
	dependencies, the error is captured and included in the results.
	
	If `model_dir` is provided, trained models are saved.
	If `predictions_dir` is provided, per-instance predictions are saved.
	"""
	print(f"\n=== Dataset: {dataset_name} ({len(DEFAULT_BENCHMARK_SPECS) + (1 if include_tsf else 0)} classifiers) ===", flush=True)
	print(f"  Loading data...", flush=True)
	X_train, y_train, X_test, y_test = load_ucr_dataset(data_dir=data_dir, dataset_name=dataset_name)
	print(f"  Loaded: train={X_train.shape}, test={X_test.shape}", flush=True)

	selected_specs = list(DEFAULT_BENCHMARK_SPECS)
	if benchmark_names is not None:
		requested = {name.strip().lower() for name in benchmark_names if name.strip()}
		selected_specs = [spec for spec in DEFAULT_BENCHMARK_SPECS if spec.name.lower() in requested]

	rows: list[dict[str, Any]] = []

	def _checkpoint_row(row: dict[str, Any]) -> None:
		if checkpoint_path is None:
			return
		try:
			_append_jsonl_row(row, Path(checkpoint_path))
		except Exception as exc:
			print(f"WARNING: unable to checkpoint row (dataset={dataset_name}): {exc}", flush=True)

	best_params_map = load_tsf_best_params(ts_config_path) if ts_config_path else {}
	selected_tsf_params = best_params_map.get(dataset_name, {
		"n_estimators": n_estimators_tsf,
		"random_state": random_state,
	})

	if include_tsf:
		try:
			tsfc = AeonTSFClassifier(
				config=_build_tsf_config_from_params(selected_tsf_params, default_random_state=random_state)
			)
			model_path = _model_path(Path(model_dir), dataset_name, "TSF (ours)") if model_dir else None
			predictions_path = _predictions_path(Path(predictions_dir), dataset_name, "TSF (ours)") if predictions_dir else None
			result = _evaluate_model(
				model=tsfc,
				model_name="TSF (ours)",
				dataset_name=dataset_name,
				X_train=X_train,
				y_train=y_train,
				X_test=X_test,
				y_test=y_test,
				model_path=model_path,
				predictions_path=predictions_path,
			)
			rows.append(result)
			_checkpoint_row(result)
		except Exception as exc:
			print(f"  [{dataset_name}] TSF (ours) — ERROR: {type(exc).__name__}: {exc}", flush=True)
			row = _error_result_row(dataset_name, "TSF (ours)", exc)
			rows.append(row)
			_checkpoint_row(row)

	for spec in selected_specs:
		try:
			model = instantiate_benchmark(spec=spec, random_state=random_state)
			model_path = _model_path(Path(model_dir), dataset_name, spec.name) if model_dir else None
			predictions_path = _predictions_path(Path(predictions_dir), dataset_name, spec.name) if predictions_dir else None
			result = _evaluate_model(
				model=model,
				model_name=spec.name,
				dataset_name=dataset_name,
				X_train=X_train,
				y_train=y_train,
				X_test=X_test,
				y_test=y_test,
				model_path=model_path,
				predictions_path=predictions_path,
			)
			rows.append(result)
			_checkpoint_row(result)
		except Exception as exc:
			print(f"  [{dataset_name}] {spec.name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
			row = _error_result_row(dataset_name, spec.name, exc)
			rows.append(row)
			_checkpoint_row(row)

	print(f"=== {dataset_name} done ===", flush=True)
	return rows


def run_train_suite(
	dataset_name: str,
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
	ts_config_path: str | Path | None = "results/best_of_best_tsf.json",
	checkpoint_path: str | Path | None = None,
	ask_on_existing_model: bool = False,
	load_existing_if_available: bool = False,
	model_dir: str | Path = "trained_models",
) -> list[dict[str, Any]]:
	"""Train classifiers on a dataset and persist the trained models.

	If `ask_on_existing_model` is True and a model file already exists, the user is
	prompted to either load the existing model (and evaluate it) or retrain without
	saving/overwriting the existing file.

	If `load_existing_if_available` is True, existing models are loaded automatically
	(without prompting), and only missing models are trained.
	"""

	load_all_mode = load_existing_if_available

	print(f"\n=== Dataset: {dataset_name} (train + save) ===", flush=True)
	print(f"  Loading data...", flush=True)
	X_train, y_train, X_test, y_test = load_ucr_dataset(data_dir=data_dir, dataset_name=dataset_name)
	print(f"  Loaded: train={X_train.shape}, test={X_test.shape}", flush=True)

	selected_specs = list(DEFAULT_BENCHMARK_SPECS)
	if benchmark_names is not None:
		requested = {name.strip().lower() for name in benchmark_names if name.strip()}
		selected_specs = [spec for spec in DEFAULT_BENCHMARK_SPECS if spec.name.lower() in requested]

	rows: list[dict[str, Any]] = []

	def _checkpoint_row(row: dict[str, Any]) -> None:
		if checkpoint_path is None:
			return
		try:
			_append_jsonl_row(row, Path(checkpoint_path))
		except Exception as exc:
			print(f"WARNING: unable to checkpoint row (dataset={dataset_name}): {exc}", flush=True)

	best_params_map = load_tsf_best_params(ts_config_path) if ts_config_path else {}
	selected_tsf_params = best_params_map.get(dataset_name, {
		"n_estimators": n_estimators_tsf,
		"random_state": random_state,
	})

	if include_tsf:
		model_name = "TSF (ours)"
		model_path = _model_path(Path(model_dir), dataset_name, model_name)
		handled = False

		if model_path.exists() and load_all_mode:
			try:
				model = load_model(model_path)
				rows.append(
					_evaluate_loaded_model(
						model=model,
						model_name=model_name,
						dataset_name=dataset_name,
						X_test=X_test,
						y_test=y_test,
					)
				)
			except Exception as exc:
				print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
				rows.append(_error_result_row(dataset_name, model_name, exc))
			handled = True

		elif model_path.exists() and ask_on_existing_model:
			action = _prompt_load_or_retrain(model_path)
			if action == "load_all":
				load_all_mode = True
				action = "load"

			if action == "load":
				try:
					model = load_model(model_path)
					rows.append(
						_evaluate_loaded_model(
							model=model,
							model_name=model_name,
							dataset_name=dataset_name,
							X_test=X_test,
							y_test=y_test,
						)
					)
				except Exception as exc:
					print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
					rows.append(_error_result_row(dataset_name, model_name, exc))
				handled = True
			elif action == "retrain":
				try:
					tsfc = AeonTSFClassifier(
						config=_build_tsf_config_from_params(selected_tsf_params, default_random_state=random_state)
					)
					rows.append(
						_evaluate_model(
							model=tsfc,
							model_name=model_name,
							dataset_name=dataset_name,
							X_train=X_train,
							y_train=y_train,
							X_test=X_test,
							y_test=y_test,
							# Do not overwrite the existing trained model.
							model_path=None,
						)
					)
				except Exception as exc:
					print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
					rows.append(_error_result_row(dataset_name, model_name, exc))
				handled = True

		if not handled:
			try:
				tsfc = AeonTSFClassifier(
					config=_build_tsf_config_from_params(selected_tsf_params, default_random_state=random_state)
				)
				rows.append(
					_evaluate_model(
						model=tsfc,
						model_name=model_name,
						dataset_name=dataset_name,
						X_train=X_train,
						y_train=y_train,
						X_test=X_test,
						y_test=y_test,
						model_path=model_path,
					)
				)
			except Exception as exc:
				print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
				rows.append(_error_result_row(dataset_name, model_name, exc))

	# === Benchmark classifiers ===
	for spec in selected_specs:
		model_name = spec.name
		model_path = _model_path(Path(model_dir), dataset_name, model_name)
		handled = False

		if model_path.exists() and load_all_mode:
			try:
				model = load_model(model_path)
				rows.append(
					_evaluate_loaded_model(
						model=model,
						model_name=model_name,
						dataset_name=dataset_name,
						X_test=X_test,
						y_test=y_test,
					)
				)
			except Exception as exc:
				print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
				rows.append(_error_result_row(dataset_name, model_name, exc))
			handled = True

		elif model_path.exists() and ask_on_existing_model:
			action = _prompt_load_or_retrain(model_path)
			if action == "load_all":
				load_all_mode = True
				action = "load"

			if action == "load":
				try:
					model = load_model(model_path)
					rows.append(
						_evaluate_loaded_model(
							model=model,
							model_name=model_name,
							dataset_name=dataset_name,
							X_test=X_test,
							y_test=y_test,
						)
					)
				except Exception as exc:
					print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
					rows.append(_error_result_row(dataset_name, model_name, exc))
				handled = True
			elif action == "retrain":
				try:
					model = instantiate_benchmark(spec=spec, random_state=random_state)
					rows.append(
						_evaluate_model(
							model=model,
							model_name=model_name,
							dataset_name=dataset_name,
							X_train=X_train,
							y_train=y_train,
							X_test=X_test,
							y_test=y_test,
							# Do not overwrite the existing trained model.
							model_path=None,
						)
					)
				except Exception as exc:
					print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
					rows.append(_error_result_row(dataset_name, model_name, exc))
				handled = True

		if not handled:
			try:
				model = instantiate_benchmark(spec=spec, random_state=random_state)
				rows.append(
					_evaluate_model(
						model=model,
						model_name=model_name,
						dataset_name=dataset_name,
						X_train=X_train,
						y_train=y_train,
						X_test=X_test,
						y_test=y_test,
						model_path=model_path,
					)
				)
			except Exception as exc:
				print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
				rows.append(_error_result_row(dataset_name, model_name, exc))

	print(f"=== {dataset_name} done ===", flush=True)
	return rows


def run_predict_suite(
	dataset_name: str,
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
	model_dir: str | Path = "trained_models",
	predictions_dir: str | Path = "results/predictions",
) -> list[dict[str, Any]]:
	"""Predict using pre-trained models and save per-instance predictions."""
	print(f"\n=== Dataset: {dataset_name} (predict) ===", flush=True)
	print(f"  Loading data...", flush=True)
	_, _, X_test, y_test = load_ucr_dataset(data_dir=data_dir, dataset_name=dataset_name)
	print(f"  Loaded: test={X_test.shape}", flush=True)

	selected_specs = list(DEFAULT_BENCHMARK_SPECS)
	if benchmark_names is not None:
		requested = {name.strip().lower() for name in benchmark_names if name.strip()}
		selected_specs = [spec for spec in DEFAULT_BENCHMARK_SPECS if spec.name.lower() in requested]

	rows: list[dict[str, Any]] = []

	if include_tsf:
		model_name = "TSF (ours)"
		model_path = _model_path(Path(model_dir), dataset_name, model_name)
		predictions_path = _predictions_path(Path(predictions_dir), dataset_name, model_name)
		try:
			model = load_model(model_path)
			rows.append(
				_evaluate_loaded_model(
					model=model,
					model_name=model_name,
					dataset_name=dataset_name,
					X_test=X_test,
					y_test=y_test,
					predictions_path=predictions_path,
				)
			)
		except Exception as exc:
			print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
			rows.append(_error_result_row(dataset_name, model_name, exc))

	for spec in selected_specs:
		model_name = spec.name
		model_path = _model_path(Path(model_dir), dataset_name, model_name)
		predictions_path = _predictions_path(Path(predictions_dir), dataset_name, model_name)
		try:
			model = load_model(model_path)
			rows.append(
				_evaluate_loaded_model(
					model=model,
					model_name=model_name,
					dataset_name=dataset_name,
					X_test=X_test,
					y_test=y_test,
					predictions_path=predictions_path,
				)
			)
		except Exception as exc:
			print(f"  [{dataset_name}] {model_name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
			rows.append(_error_result_row(dataset_name, model_name, exc))

	print(f"=== {dataset_name} done ===", flush=True)
	return rows


# Backwards compatibility: old name
run_forecast_suite = run_predict_suite


def run_benchmarks_on_datasets(
	datasets: list[str],
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
	ts_config_path: str | Path | None = "results/best_of_best_tsf.json",
	model_dir: str | Path | None = None,
	predictions_dir: str | Path | None = None,
	checkpoint_dir: str | Path | None = None,
	n_jobs: int = 1,
) -> list[dict[str, Any]]:
	"""Run benchmark suite across multiple datasets."""
	print(f"Starting benchmark run: {len(datasets)} dataset(s) × up to {len(DEFAULT_BENCHMARK_SPECS) + (1 if include_tsf else 0)} classifiers", flush=True)

	if n_jobs == 1:
		all_rows: list[dict[str, Any]] = []
		for i, dataset_name in enumerate(datasets, 1):
			print(f"\n[{i}/{len(datasets)}]", end=" ", flush=True)
			rows = run_benchmark_suite(
				dataset_name=dataset_name,
				data_dir=data_dir,
				benchmark_names=benchmark_names,
				include_tsf=include_tsf,
				random_state=random_state,
				n_estimators_tsf=n_estimators_tsf,
				ts_config_path=ts_config_path,
				model_dir=model_dir,
				predictions_dir=predictions_dir,
				checkpoint_path=_dataset_checkpoint_path(checkpoint_dir, dataset_name),
			)
			all_rows.extend(rows)
		return all_rows

	results = Parallel(n_jobs=n_jobs)(
		delayed(run_benchmark_suite)(
			dataset_name=dataset_name,
			data_dir=data_dir,
			benchmark_names=benchmark_names,
			include_tsf=include_tsf,
			random_state=random_state,
			n_estimators_tsf=n_estimators_tsf,
			ts_config_path=ts_config_path,
			model_dir=model_dir,
			predictions_dir=predictions_dir,
			checkpoint_path=_dataset_checkpoint_path(checkpoint_dir, dataset_name),
		)
		for dataset_name in datasets
	)

	all_rows: list[dict[str, Any]] = []
	for part in results:
		all_rows.extend(part)

	return all_rows


def run_train_on_datasets(
	datasets: list[str],
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
	ts_config_path: str | Path | None = "results/best_of_best_tsf.json",
	ask_on_existing_model: bool = False,
	load_existing_if_available: bool = False,
	model_dir: str | Path = "trained_models",
	checkpoint_dir: str | Path | None = None,
	n_jobs: int = 1,
) -> list[dict[str, Any]]:
	"""Train models across datasets in parallel optionally."""
	print(f"Starting train suite: {len(datasets)} dataset(s) × up to {len(DEFAULT_BENCHMARK_SPECS) + (1 if include_tsf else 0)} classifiers", flush=True)

	if n_jobs == 1:
		all_rows = []
		for i, dataset_name in enumerate(datasets, 1):
			print(f"\n[{i}/{len(datasets)}]", end=" ", flush=True)
			rows = run_train_suite(
				dataset_name=dataset_name,
				data_dir=data_dir,
				benchmark_names=benchmark_names,
				include_tsf=include_tsf,
				random_state=random_state,
				n_estimators_tsf=n_estimators_tsf,
				ts_config_path=ts_config_path,
				ask_on_existing_model=ask_on_existing_model,
				load_existing_if_available=load_existing_if_available,
				model_dir=model_dir,
				checkpoint_path=_dataset_checkpoint_path(checkpoint_dir, dataset_name),
			)
			all_rows.extend(rows)
		return all_rows

	results = Parallel(n_jobs=n_jobs)(
		delayed(run_train_suite)(
			dataset_name=dataset_name,
			data_dir=data_dir,
			benchmark_names=benchmark_names,
			include_tsf=include_tsf,
			random_state=random_state,
			n_estimators_tsf=n_estimators_tsf,
			ts_config_path=ts_config_path,
			ask_on_existing_model=ask_on_existing_model,
			load_existing_if_available=load_existing_if_available,
			model_dir=model_dir,
			checkpoint_path=_dataset_checkpoint_path(checkpoint_dir, dataset_name),
		)
		for dataset_name in datasets
	)

	all_rows: list[dict[str, Any]] = []
	for part in results:
		all_rows.extend(part)

	return all_rows


def save_results_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
	"""Save result rows to CSV.

	Automatically includes any extra fields that appear in the rows (e.g., model_path,
	predictions_path) so the output captures all recorded metadata.
	"""
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)

	base_fields = [
		"dataset",
		"classifier",
		"accuracy",
		"precision_weighted",
		"recall_weighted",
		"f1_weighted",
		"balanced_accuracy",
		"fit_time_s",
		"predict_time_s",
		"status",
		"error",
	]
	extra_fields = sorted(
		{k for row in rows for k in row.keys() if k not in base_fields}
	)
	fieldnames = base_fields + extra_fields

	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	return path


def format_results_table(rows: list[dict[str, Any]]) -> str:
	"""Create a compact plain-text table for terminal output."""
	if not rows:
		return "No results."

	headers = [
		"dataset",
		"classifier",
		"accuracy",
		"f1_weighted",
		"balanced_accuracy",
		"fit_time_s",
		"predict_time_s",
		"status",
	]
	col_widths: dict[str, int] = {}
	for header in headers:
		max_cell_len = max(len(str(row.get(header, ""))) for row in rows)
		col_widths[header] = max(len(header), max_cell_len)

	def _fmt_cell(header: str, value: Any) -> str:
		if header in {"accuracy", "f1_weighted", "balanced_accuracy", "fit_time_s", "predict_time_s"} and isinstance(value, (float, np.floating)):
			if np.isnan(value):
				return "nan"
			return f"{value:.4f}"
		return str(value)

	line = " | ".join(header.ljust(col_widths[header]) for header in headers)
	sep = "-+-".join("-" * col_widths[header] for header in headers)
	data_lines = []
	for row in rows:
		formatted = []
		for header in headers:
			formatted.append(_fmt_cell(header, row.get(header, "")).ljust(col_widths[header]))
		data_lines.append(" | ".join(formatted))

	return "\n".join([line, sep, *data_lines])


def compute_wilcoxon_vs_reference(
	rows: list[dict[str, Any]],
	reference_classifier: str = "TSF (ours)",
	metric: str = "accuracy",
	alpha: float = 0.05,
) -> list[dict[str, Any]]:
	"""Compute paired Wilcoxon tests versus a reference classifier across datasets."""
	if not rows:
		return []

	# Keep one metric value per (dataset, classifier) and only successful runs.
	values: dict[tuple[str, str], float] = {}
	for row in rows:
		if row.get("status") != "ok":
			continue
		dataset = row.get("dataset")
		classifier = row.get("classifier")
		value = row.get(metric)
		if dataset is None or classifier is None:
			continue
		if not isinstance(value, (int, float, np.floating)):
			continue
		value_float = float(value)
		if np.isnan(value_float):
			continue
		values[(str(dataset), str(classifier))] = value_float

	datasets = sorted({dataset for dataset, _ in values.keys()})
	classifiers = sorted({classifier for _, classifier in values.keys() if classifier != reference_classifier})

	results: list[dict[str, Any]] = []
	for classifier in classifiers:
		reference_scores: list[float] = []
		candidate_scores: list[float] = []
		for dataset in datasets:
			ref_key = (dataset, reference_classifier)
			cmp_key = (dataset, classifier)
			if ref_key in values and cmp_key in values:
				reference_scores.append(values[ref_key])
				candidate_scores.append(values[cmp_key])

		if not reference_scores:
			continue

		ref_arr = np.asarray(reference_scores, dtype=float)
		cmp_arr = np.asarray(candidate_scores, dtype=float)
		delta_arr = ref_arr - cmp_arr

		better = int(np.sum(delta_arr > 0))
		worse = int(np.sum(delta_arr < 0))
		ties = int(np.sum(delta_arr == 0))

		row_result: dict[str, Any] = {
			"reference_classifier": reference_classifier,
			"classifier": classifier,
			"metric": metric,
			"n_pairs": int(ref_arr.size),
			"reference_mean": float(np.mean(ref_arr)),
			"candidate_mean": float(np.mean(cmp_arr)),
			"mean_delta": float(np.mean(delta_arr)),
			"median_delta": float(np.median(delta_arr)),
			"reference_better_count": better,
			"candidate_better_count": worse,
			"ties_count": ties,
			"alpha": alpha,
			"significant": False,
			"wilcoxon_stat": np.nan,
			"p_value": np.nan,
			"status": "ok",
			"error": "",
		}

		if np.all(delta_arr == 0.0):
			row_result["status"] = "all_ties"
			results.append(row_result)
			continue

		try:
			stat, p_value = wilcoxon(ref_arr, cmp_arr, alternative="two-sided")
			row_result["wilcoxon_stat"] = float(stat)
			row_result["p_value"] = float(p_value)
			row_result["significant"] = bool(p_value < alpha)
		except ValueError as exc:
			row_result["status"] = "error"
			row_result["error"] = f"{type(exc).__name__}: {exc}"

		results.append(row_result)

	return results


def save_wilcoxon_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
	"""Save Wilcoxon comparison rows to CSV."""
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)

	if not rows:
		with path.open("w", newline="", encoding="utf-8") as handle:
			writer = csv.writer(handle)
			writer.writerow(
				[
					"reference_classifier",
					"classifier",
					"metric",
					"n_pairs",
					"reference_mean",
					"candidate_mean",
					"mean_delta",
					"median_delta",
					"reference_better_count",
					"candidate_better_count",
					"ties_count",
					"alpha",
					"significant",
					"wilcoxon_stat",
					"p_value",
					"status",
					"error",
				]
			)
		return path

	fieldnames = [
		"reference_classifier",
		"classifier",
		"metric",
		"n_pairs",
		"reference_mean",
		"candidate_mean",
		"mean_delta",
		"median_delta",
		"reference_better_count",
		"candidate_better_count",
		"ties_count",
		"alpha",
		"significant",
		"wilcoxon_stat",
		"p_value",
		"status",
		"error",
	]

	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	return path


def format_wilcoxon_table(rows: list[dict[str, Any]]) -> str:
	"""Create a compact plain-text Wilcoxon summary table."""
	if not rows:
		return "No Wilcoxon results."

	headers = [
		"classifier",
		"n_pairs",
		"mean_delta",
		"p_value",
		"significant",
		"reference_better_count",
		"candidate_better_count",
		"ties_count",
		"status",
	]

	def _fmt_cell(header: str, value: Any) -> str:
		if header in {"mean_delta", "p_value"} and isinstance(value, (float, np.floating)):
			if np.isnan(value):
				return "nan"
			return f"{value:.4f}"
		return str(value)

	col_widths: dict[str, int] = {}
	for header in headers:
		max_cell_len = max(len(_fmt_cell(header, row.get(header, ""))) for row in rows)
		col_widths[header] = max(len(header), max_cell_len)

	line = " | ".join(header.ljust(col_widths[header]) for header in headers)
	sep = "-+-".join("-" * col_widths[header] for header in headers)
	data_lines = []
	for row in rows:
		formatted = []
		for header in headers:
			formatted.append(_fmt_cell(header, row.get(header, "")).ljust(col_widths[header]))
		data_lines.append(" | ".join(formatted))

	return "\n".join([line, sep, *data_lines])
