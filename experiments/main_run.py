"""Experiment runner for TSF and benchmark comparisons."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Suppress TensorFlow informational/warning logs (errors still show).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from classifiers.tsf_classifier import AeonTSFClassifier, TSFConfig
from classifiers.benchmarks.suite import DEFAULT_BENCHMARK_SPECS
from experiments.validation import (
	format_results_table,
	run_benchmarks_on_datasets,
	run_train_suite,
	run_predict_suite,
	# Keep old name for backward compatibility
	run_forecast_suite,
	save_results_csv,
)


def generate_synthetic_data(
	n_train: int, n_test: int, n_timestamps: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Generate a simple binary synthetic time-series classification dataset."""
	rng = np.random.default_rng(seed)
	X_train = rng.normal(size=(n_train, n_timestamps))
	X_test = rng.normal(size=(n_test, n_timestamps))

	y_train = (X_train.mean(axis=1) > 0.0).astype(int)
	y_test = (X_test.mean(axis=1) > 0.0).astype(int)
	return X_train, y_train, X_test, y_test


def _parse_csv_list(raw: str) -> list[str]:
	"""Parse comma-separated values into a cleaned list."""
	return [item.strip() for item in raw.split(",") if item.strip()]


def _sanitize_filename(name: str) -> str:
	"""Make a string safe to use in a filename."""
	return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def _resolve_output_path(output_template: str, dataset: str, multiple_datasets: bool) -> str:
	"""Resolve an output CSV path for a given dataset.

	If the template contains '{dataset}', it is substituted.
	If multiple datasets are being processed (or the template is the default), the dataset name is inserted before the file extension.
	"""
	if "{dataset}" in output_template:
		return output_template.format(dataset=_sanitize_filename(dataset))

	if multiple_datasets:
		path = Path(output_template)
		stem = path.stem
		suffix = path.suffix
		return str(path.with_name(f"{stem}_{_sanitize_filename(dataset)}{suffix}"))

	return output_template


def main() -> None:
	parser = argparse.ArgumentParser(description="Run TSF experiments and benchmark comparisons.")
	parser.add_argument(
		"--mode",
		choices=["benchmarks", "synthetic", "train", "predict", "forecast"],
		default="benchmarks",
		help=(
			"Execution mode. 'benchmarks' compares TSF vs benchmark algorithms on UCR data; "
			"'train' trains models and saves them; 'predict' (alias 'forecast') loads saved models and runs prediction; "
			"'synthetic' runs a quick synthetic TSF experiment."
		),
	)
	parser.add_argument("--seed", type=int, default=42)

	# Synthetic mode arguments.
	parser.add_argument("--n-train", type=int, default=60)
	parser.add_argument("--n-test", type=int, default=20)
	parser.add_argument("--n-timestamps", type=int, default=80)
	parser.add_argument("--n-estimators", type=int, default=200)

	# Benchmark mode arguments.
	parser.add_argument(
		"--datasets",
		type=str,
		default="ItalyPowerDemand",
		help="Comma-separated datasets under data/ (e.g. ItalyPowerDemand,GunPoint,ECG5000).",
	)
	parser.add_argument(
		"--benchmarks",
		type=str,
		default="",
		help=(
			"Optional comma-separated subset of benchmark names. "
			"If omitted, runs all defaults: "
			+ ", ".join(spec.name for spec in DEFAULT_BENCHMARK_SPECS)
		),
	)
	parser.add_argument(
		"--data-dir",
		type=str,
		default="data",
		help="Directory containing UCR dataset folders.",
	)
	parser.add_argument(
		"--output-csv",
		type=str,
		default="results/benchmark_comparison.csv",
		help="Path to write benchmark comparison CSV output.",
	)
	parser.add_argument(
		"--model-dir",
		type=str,
		default="trained_models",
		help="Directory where trained models are saved/loaded.",
	)
	parser.add_argument(
		"--predictions-dir",
		type=str,
		default="results/predictions",
		help="Directory where per-instance prediction CSVs are written during forecasting.",
	)
	parser.add_argument(
		"--no-tsf",
		action="store_true",
		help="Exclude TSF (ours) from benchmark runs.",
	)
	args = parser.parse_args()

	if args.mode == "synthetic":
		X_train, y_train, X_test, y_test = generate_synthetic_data(
			n_train=args.n_train,
			n_test=args.n_test,
			n_timestamps=args.n_timestamps,
			seed=args.seed,
		)

		config = TSFConfig(n_estimators=args.n_estimators, random_state=args.seed)
		clf = AeonTSFClassifier(config=config)
		clf.fit(X_train, y_train)
		accuracy = clf.score(X_test, y_test)
		print(f"TSF accuracy (synthetic): {accuracy:.3f}")
		return

	datasets = _parse_csv_list(args.datasets)
	benchmark_names = _parse_csv_list(args.benchmarks) if args.benchmarks else None

	# Resolve output file naming to avoid overwriting when running with multiple datasets.
	# If the template includes {dataset}, it will be substituted.
	# Otherwise, when using the default output path, we insert the dataset name to keep runs separate.
	use_dataset_in_name = len(datasets) > 1 or args.output_csv == "results/benchmark_comparison.csv"
	output_path = _resolve_output_path(args.output_csv, datasets[0] if datasets else "", use_dataset_in_name)

	if args.mode == "benchmarks":
		rows = run_benchmarks_on_datasets(
			datasets=datasets,
			data_dir=args.data_dir,
			benchmark_names=benchmark_names,
			include_tsf=not args.no_tsf,
			random_state=args.seed,
			n_estimators_tsf=args.n_estimators,
		)

		print(format_results_table(rows))
		# Use a combined output file for all datasets (default or explicitly provided)
		combined_output_path = (
			args.output_csv
			if len(datasets) == 1
			else _resolve_output_path(args.output_csv, "all", True)
		)
		combined_output_path = save_results_csv(rows=rows, output_path=combined_output_path)
		print(f"\nSaved combined results to: {combined_output_path}")

		# Also save per-dataset files when multiple datasets are present
		if len(datasets) > 1:
			for ds in datasets:
				per_dataset_path = _resolve_output_path(args.output_csv, ds, True)
				per_rows = [r for r in rows if r.get("dataset") == ds]
				save_results_csv(rows=per_rows, output_path=per_dataset_path)
				print(f"Saved per-dataset results to: {per_dataset_path}")

	elif args.mode == "train":
		rows = []
		for dataset in datasets:
			rows.extend(
				run_train_suite(
					dataset_name=dataset,
					data_dir=args.data_dir,
					benchmark_names=benchmark_names,
					include_tsf=not args.no_tsf,
					random_state=args.seed,
					n_estimators_tsf=args.n_estimators,
					model_dir=args.model_dir,
				)
			)
		for dataset in datasets:
			per_dataset_path = _resolve_output_path(args.output_csv, dataset, True)
			per_rows = [r for r in rows if r.get("dataset") == dataset]
			save_results_csv(rows=per_rows, output_path=per_dataset_path)
			print(f"Saved results to: {per_dataset_path}")

	elif args.mode in ("predict", "forecast"):
		rows = []
		for dataset in datasets:
			rows.extend(
				run_predict_suite(
					dataset_name=dataset,
					data_dir=args.data_dir,
					benchmark_names=benchmark_names,
					include_tsf=not args.no_tsf,
					random_state=args.seed,
					n_estimators_tsf=args.n_estimators,
					model_dir=args.model_dir,
					predictions_dir=args.predictions_dir,
				)
			)
		for dataset in datasets:
			per_dataset_path = _resolve_output_path(args.output_csv, dataset, True)
			per_rows = [r for r in rows if r.get("dataset") == dataset]
			save_results_csv(rows=per_rows, output_path=per_dataset_path)
			print(f"Saved results to: {per_dataset_path}")

	else:
		# This should not happen because argparse validates the choice
		raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
	main()
