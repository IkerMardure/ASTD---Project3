"""Validation utilities for comparing TS classifiers on UCR datasets.

This module defines a default benchmark suite and a small evaluation pipeline
to compare classifiers with consistent metrics.
"""

from __future__ import annotations

import csv
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np

from classifiers.benchmarks.suite import (
	DEFAULT_BENCHMARK_SPECS,
	instantiate_benchmark,
)
from classifiers.tsf_classifier import AeonTSFClassifier, TSFConfig


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


def _resolve_tsf_config(
	dataset_name: str,
	tsf_params: dict[str, dict[str, int]] | None,
	default_n_estimators: int,
	default_min_interval: int,
	random_state: int | None,
) -> TSFConfig:
	if not tsf_params:
		return TSFConfig(n_estimators=default_n_estimators, min_interval_length=default_min_interval, random_state=random_state)
	lookup = dataset_name if dataset_name in tsf_params else next((k for k in tsf_params.keys() if k.lower() == dataset_name.lower()), None)
	values = tsf_params.get(lookup, {}) if lookup is not None else {}
	return TSFConfig(
		n_estimators=values.get("n_estimators", default_n_estimators),
		min_interval_length=values.get("min_interval_length", default_min_interval),
		random_state=random_state,
	)


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

	accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
	print(f"  [{dataset_name}] {model_name} — accuracy={accuracy:.4f}, predict={predict_time_s:.1f}s", flush=True)

	row: dict[str, Any] = {
		"dataset": dataset_name,
		"classifier": model_name,
		"accuracy": accuracy,
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

	accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
	print(f"  [{dataset_name}] {model_name} — accuracy={accuracy:.4f}, predict={predict_time_s:.1f}s", flush=True)

	row: dict[str, Any] = {
		"dataset": dataset_name,
		"classifier": model_name,
		"accuracy": accuracy,
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
	tsf_config: TSFConfig | None = None,
) -> list[dict[str, Any]]:
	"""Run the default benchmark suite on one dataset.

	If a model cannot be instantiated or trained due to missing optional
	dependencies, the error is captured and included in the results.
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

	if include_tsf:
		try:
			cfg = tsf_config or TSFConfig(n_estimators=n_estimators_tsf, random_state=random_state)
			tsfc = AeonTSFClassifier(config=cfg)
			rows.append(
				_evaluate_model(
					model=tsfc,
					model_name="TSF (ours)",
					dataset_name=dataset_name,
					X_train=X_train,
					y_train=y_train,
					X_test=X_test,
					y_test=y_test,
				)
			)
		except Exception as exc:
			print(f"  [{dataset_name}] TSF (ours) — ERROR: {type(exc).__name__}: {exc}", flush=True)
			rows.append(
				{
					"dataset": dataset_name,
					"classifier": "TSF (ours)",
					"accuracy": np.nan,
					"fit_time_s": np.nan,
					"predict_time_s": np.nan,
					"status": "error",
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

	for spec in selected_specs:
		try:
			model = instantiate_benchmark(spec=spec, random_state=random_state)
			rows.append(
				_evaluate_model(
					model=model,
					model_name=spec.name,
					dataset_name=dataset_name,
					X_train=X_train,
					y_train=y_train,
					X_test=X_test,
					y_test=y_test,
				)
			)
		except Exception as exc:
			print(f"  [{dataset_name}] {spec.name} — ERROR: {type(exc).__name__}: {exc}", flush=True)
			rows.append(
				{
					"dataset": dataset_name,
					"classifier": spec.name,
					"accuracy": np.nan,
					"fit_time_s": np.nan,
					"predict_time_s": np.nan,
					"status": "error",
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

	print(f"=== {dataset_name} done ===", flush=True)
	return rows


def run_train_suite(
	dataset_name: str,
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
	tsf_config: TSFConfig | None = None,
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
				rows.append(
					{
						"dataset": dataset_name,
						"classifier": model_name,
						"accuracy": np.nan,
						"fit_time_s": np.nan,
						"predict_time_s": np.nan,
						"status": "error",
						"error": f"{type(exc).__name__}: {exc}",
					}
				)
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
					rows.append(
						{
							"dataset": dataset_name,
							"classifier": model_name,
							"accuracy": np.nan,
							"fit_time_s": np.nan,
							"predict_time_s": np.nan,
							"status": "error",
							"error": f"{type(exc).__name__}: {exc}",
						}
					)
				handled = True
			elif action == "retrain":
				try:
					cfg = tsf_config or TSFConfig(n_estimators=n_estimators_tsf, random_state=random_state)
					tsfc = AeonTSFClassifier(config=cfg)
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
					rows.append(
						{
							"dataset": dataset_name,
							"classifier": model_name,
							"accuracy": np.nan,
							"fit_time_s": np.nan,
							"predict_time_s": np.nan,
							"status": "error",
							"error": f"{type(exc).__name__}: {exc}",
						}
					)
				handled = True

		if not handled:
			try:
				cfg = tsf_config or TSFConfig(n_estimators=n_estimators_tsf, random_state=random_state)
				tsfc = AeonTSFClassifier(config=cfg)
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
				rows.append(
					{
						"dataset": dataset_name,
						"classifier": model_name,
						"accuracy": np.nan,
						"fit_time_s": np.nan,
						"predict_time_s": np.nan,
						"status": "error",
						"error": f"{type(exc).__name__}: {exc}",
					}
				)

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
				rows.append(
					{
						"dataset": dataset_name,
						"classifier": model_name,
						"accuracy": np.nan,
						"fit_time_s": np.nan,
						"predict_time_s": np.nan,
						"status": "error",
						"error": f"{type(exc).__name__}: {exc}",
					}
				)
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
					rows.append(
						{
							"dataset": dataset_name,
							"classifier": model_name,
							"accuracy": np.nan,
							"fit_time_s": np.nan,
							"predict_time_s": np.nan,
							"status": "error",
							"error": f"{type(exc).__name__}: {exc}",
						}
					)
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
					rows.append(
						{
							"dataset": dataset_name,
							"classifier": model_name,
							"accuracy": np.nan,
							"fit_time_s": np.nan,
							"predict_time_s": np.nan,
							"status": "error",
							"error": f"{type(exc).__name__}: {exc}",
						}
					)
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
				rows.append(
					{
						"dataset": dataset_name,
						"classifier": model_name,
						"accuracy": np.nan,
						"fit_time_s": np.nan,
						"predict_time_s": np.nan,
						"status": "error",
						"error": f"{type(exc).__name__}: {exc}",
					}
				)

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
			rows.append(
				{
					"dataset": dataset_name,
					"classifier": model_name,
					"accuracy": np.nan,
					"fit_time_s": np.nan,
					"predict_time_s": np.nan,
					"status": "error",
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

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
			rows.append(
				{
					"dataset": dataset_name,
					"classifier": model_name,
					"accuracy": np.nan,
					"fit_time_s": np.nan,
					"predict_time_s": np.nan,
					"status": "error",
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

	print(f"=== {dataset_name} done ===", flush=True)
	return rows


# Backwards compatibility: old name
run_forecast_suite = run_predict_suite


def _benchmark_dataset_task(args: tuple[str, str | Path, list[str] | None, bool, int | None, int, dict[str, dict[str, int]] | None]) -> list[dict[str, Any]]:
	"""Module-level helper for pickle-safe multiprocessing benchmark task."""
	dataset_name, data_dir, benchmark_names, include_tsf, random_state, n_estimators_tsf, tsf_params = args
	print(f"\n[dataset {dataset_name}]", flush=True)
	dataset_tsf_config = _resolve_tsf_config(
		dataset_name=dataset_name,
		tsf_params=tsf_params,
		default_n_estimators=n_estimators_tsf,
		default_min_interval=TSFConfig().min_interval_length,
		random_state=random_state,
	)
	return run_benchmark_suite(
		dataset_name=dataset_name,
		data_dir=data_dir,
		benchmark_names=benchmark_names,
		include_tsf=include_tsf,
		random_state=random_state,
		n_estimators_tsf=n_estimators_tsf,
		tsf_config=dataset_tsf_config,
	)


def run_benchmarks_on_datasets(
	datasets: list[str],
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
	tsf_params: dict[str, dict[str, int]] | None = None,
	jobs: int = 1,
) -> list[dict[str, Any]]:
	"""Run benchmark suite across multiple datasets."""
	print(f"Starting benchmark run: {len(datasets)} dataset(s) × up to {len(DEFAULT_BENCHMARK_SPECS) + (1 if include_tsf else 0)} classifiers", flush=True)

	all_rows: list[dict[str, Any]] = []
	if jobs is None or jobs <= 1:
		for dataset_name in datasets:
			all_rows.extend(_benchmark_dataset_task((dataset_name, data_dir, benchmark_names, include_tsf, random_state, n_estimators_tsf, tsf_params)))
	else:
		worker_count = min(jobs, cpu_count())
		print(f"Running benchmarks in parallel with {worker_count} workers.")
		with Pool(processes=worker_count) as pool:
			args_list = [
				(dataset_name, data_dir, benchmark_names, include_tsf, random_state, n_estimators_tsf, tsf_params)
				for dataset_name in datasets
			]
			for dataset_rows in pool.imap_unordered(_benchmark_dataset_task, args_list):
				all_rows.extend(dataset_rows)

	return all_rows


def _train_dataset_task(args: tuple[str, str | Path, list[str] | None, bool, int | None, int, dict[str, dict[str, int]] | None, bool, bool, str | Path]) -> list[dict[str, Any]]:
	"""Module-level helper for pickle-safe multiprocessing train task."""
	dataset_name, data_dir, benchmark_names, include_tsf, random_state, n_estimators_tsf, tsf_params, ask_on_existing_model, load_existing_if_available, model_dir = args
	print(f"\n[dataset {dataset_name}] train", flush=True)
	dataset_tsf_config = _resolve_tsf_config(
		dataset_name=dataset_name,
		tsf_params=tsf_params,
		default_n_estimators=n_estimators_tsf,
		default_min_interval=TSFConfig().min_interval_length,
		random_state=random_state,
	)
	return run_train_suite(
		dataset_name=dataset_name,
		data_dir=data_dir,
		benchmark_names=benchmark_names,
		include_tsf=include_tsf,
		random_state=random_state,
		n_estimators_tsf=n_estimators_tsf,
		tsf_config=dataset_tsf_config,
		ask_on_existing_model=ask_on_existing_model,
		load_existing_if_available=load_existing_if_available,
		model_dir=model_dir,
	)


def run_train_on_datasets(
	datasets: list[str],
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
	tsf_params: dict[str, dict[str, int]] | None = None,
	ask_on_existing_model: bool = False,
	load_existing_if_available: bool = False,
	model_dir: str | Path = "trained_models",
	jobs: int = 1,
) -> list[dict[str, Any]]:
	"""Train models for multiple datasets, optionally in parallel."""
	def _train_task(dataset_name: str) -> list[dict[str, Any]]:
		print(f"\n[dataset {dataset_name}] train", flush=True)
		dataset_tsf_config = _resolve_tsf_config(
			dataset_name=dataset_name,
			tsf_params=tsf_params,
			default_n_estimators=n_estimators_tsf,
			default_min_interval=TSFConfig().min_interval_length,
			random_state=random_state,
		)
		return run_train_suite(
			dataset_name=dataset_name,
			data_dir=data_dir,
			benchmark_names=benchmark_names,
			include_tsf=include_tsf,
			random_state=random_state,
			n_estimators_tsf=n_estimators_tsf,
			tsf_config=dataset_tsf_config,
			ask_on_existing_model=ask_on_existing_model,
			load_existing_if_available=load_existing_if_available,
			model_dir=model_dir,
		)

	all_rows: list[dict[str, Any]] = []
	if jobs is None or jobs <= 1:
		for dataset_name in datasets:
			all_rows.extend(_train_dataset_task((dataset_name,data_dir,benchmark_names,include_tsf,random_state,n_estimators_tsf,tsf_params,ask_on_existing_model,load_existing_if_available,model_dir)))
	else:
		worker_count = min(jobs, cpu_count())
		print(f"Running train in parallel with {worker_count} workers.")
		with Pool(processes=worker_count) as pool:
			args_list = [
				(dataset_name,data_dir,benchmark_names,include_tsf,random_state,n_estimators_tsf,tsf_params,ask_on_existing_model,load_existing_if_available,model_dir)
				for dataset_name in datasets
			]
			for dataset_rows in pool.imap_unordered(_train_dataset_task, args_list):
				all_rows.extend(dataset_rows)

	return all_rows


def run_predict_on_datasets(
	datasets: list[str],
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
	model_dir: str | Path = "trained_models",
	predictions_dir: str | Path = "results/predictions",
	jobs: int = 1,
) -> list[dict[str, Any]]:
	"""Run prediction for multiple datasets, optionally in parallel."""
	def _predict_task(dataset_name: str) -> list[dict[str, Any]]:
		print(f"\n[dataset {dataset_name}] predict", flush=True)
		return run_predict_suite(
			dataset_name=dataset_name,
			data_dir=data_dir,
			benchmark_names=benchmark_names,
			include_tsf=include_tsf,
			random_state=random_state,
			n_estimators_tsf=n_estimators_tsf,
			model_dir=model_dir,
			predictions_dir=predictions_dir,
		)

	all_rows: list[dict[str, Any]] = []
	if jobs is None or jobs <= 1:
		for dataset_name in datasets:
			all_rows.extend(_predict_task(dataset_name))
	else:
		worker_count = min(jobs, cpu_count())
		print(f"Running predict in parallel with {worker_count} workers.")
		with Pool(processes=worker_count) as pool:
			for dataset_rows in pool.imap_unordered(_predict_task, datasets):
				all_rows.extend(dataset_rows)

	return all_rows


def save_results_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
	"""Save result rows to CSV.

	Automatically includes any extra fields that appear in the rows (e.g., model_path,
	predictions_path) so the output captures all recorded metadata.
	"""
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)

	base_fields = ["dataset", "classifier", "accuracy", "fit_time_s", "predict_time_s", "status", "error"]
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

	headers = ["dataset", "classifier", "accuracy", "fit_time_s", "predict_time_s", "status"]
	col_widths: dict[str, int] = {}
	for header in headers:
		max_cell_len = max(len(str(row.get(header, ""))) for row in rows)
		col_widths[header] = max(len(header), max_cell_len)

	def _fmt_cell(header: str, value: Any) -> str:
		if header in {"accuracy", "fit_time_s", "predict_time_s"} and isinstance(value, (float, np.floating)):
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
