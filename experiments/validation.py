"""Validation utilities for comparing TS classifiers on UCR datasets.

This module defines a default benchmark suite and a small evaluation pipeline
to compare classifiers with consistent metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any

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
) -> dict[str, Any]:
	"""Fit, predict and score one model, returning metrics."""
	X_train_3d = _to_3d_numpy(X_train)
	X_test_3d = _to_3d_numpy(X_test)

	print(f"  [{dataset_name}] {model_name} — fitting on {X_train_3d.shape[0]} samples...", flush=True)
	start_fit = time.perf_counter()
	model.fit(X_train_3d, y_train)
	fit_time_s = time.perf_counter() - start_fit
	print(f"  [{dataset_name}] {model_name} — fit done ({fit_time_s:.1f}s). Predicting...", flush=True)

	start_pred = time.perf_counter()
	y_pred = model.predict(X_test_3d)
	predict_time_s = time.perf_counter() - start_pred

	accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
	print(f"  [{dataset_name}] {model_name} — accuracy={accuracy:.4f}, predict={predict_time_s:.1f}s", flush=True)

	return {
		"dataset": dataset_name,
		"classifier": model_name,
		"accuracy": accuracy,
		"fit_time_s": fit_time_s,
		"predict_time_s": predict_time_s,
		"status": "ok",
		"error": "",
	}


def run_benchmark_suite(
	dataset_name: str,
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
) -> list[dict[str, Any]]:
	"""Run the default benchmark suite on one dataset.

	If a model cannot be instantiated or trained due to missing optional
	dependencies, the error is captured and included in the results.
	"""
	n_classifiers = len(DEFAULT_BENCHMARK_SPECS if benchmark_names is None else [
		s for s in DEFAULT_BENCHMARK_SPECS if s.name.lower() in {n.strip().lower() for n in benchmark_names}
	]) + (1 if include_tsf else 0)
	print(f"\n=== Dataset: {dataset_name} ({n_classifiers} classifiers) ===", flush=True)
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
			tsf = AeonTSFClassifier(
				config=TSFConfig(
					n_estimators=n_estimators_tsf,
					random_state=random_state,
				)
			)
			rows.append(
				_evaluate_model(
					model=tsf,
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


def run_benchmarks_on_datasets(
	datasets: list[str],
	data_dir: str | Path = "data",
	benchmark_names: list[str] | None = None,
	include_tsf: bool = True,
	random_state: int | None = 42,
	n_estimators_tsf: int = 200,
) -> list[dict[str, Any]]:
	"""Run benchmark suite across multiple datasets."""
	print(f"Starting benchmark run: {len(datasets)} dataset(s) × up to {len(DEFAULT_BENCHMARK_SPECS) + (1 if include_tsf else 0)} classifiers", flush=True)
	all_rows: list[dict[str, Any]] = []
	for i, dataset_name in enumerate(datasets, 1):
		print(f"\n[{i}/{len(datasets)}]", end=" ", flush=True)
		rows = run_benchmark_suite(
			dataset_name=dataset_name,  # noqa: E501
			data_dir=data_dir,
			benchmark_names=benchmark_names,
			include_tsf=include_tsf,
			random_state=random_state,
			n_estimators_tsf=n_estimators_tsf,
		)
		all_rows.extend(rows)
	return all_rows


def save_results_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
	"""Save benchmark result rows to CSV."""
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = ["dataset", "classifier", "accuracy", "fit_time_s", "predict_time_s", "status", "error"]

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
