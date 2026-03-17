"""Utilities for visualizing predictions produced by `experiments/validation.py`.

This module provides helpers to load a predictions CSV (columns: index,y_true,y_pred)
and to plot:
  - an overlay of time series colored by correctness (per class)
  - a confusion matrix.

It mirrors the style and save conventions of `utils/visualize_TS.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union
import csv

try:
	import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
	plt = None
	_matplotlib_import_error = exc

try:
	import numpy as np
except ImportError as exc:  # pragma: no cover
	np = None
	_numpy_import_error = exc


def _ensure_dir(path: Union[str, Path]) -> Path:
	"""Ensure that a directory exists and return a Path object."""
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def _ensure_matplotlib() -> None:
	if plt is None:
		raise ImportError(
			"matplotlib is required to use visualization helpers. "
			"Install it with: pip install matplotlib"
		) from _matplotlib_import_error


def _ensure_numpy() -> None:
	if np is None:
		raise ImportError(
			"numpy is required to use visualization helpers. "
			"Install it with: pip install numpy"
		) from _numpy_import_error


def _sanitize_filename(name: str) -> str:
	"""Sanitize a string for use in filenames."""
	return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))


def _safe_dataset_name(dataset_name: Optional[str]) -> str:
	"""Return a safe (filesystem-friendly) dataset name.

	If dataset_name is None or empty, returns a sensible default.
	"""
	if not dataset_name:
		return "dataset"
	return _sanitize_filename(dataset_name)


def _get_preferred_style() -> str:
	"""Return an available matplotlib style for consistent plots."""
	candidates = ["seaborn-darkgrid", "seaborn", "ggplot", "tableau-colorblind10", "classic"]
	available = set(plt.style.available) if plt is not None else set()
	for name in candidates:
		if name in available:
			return name
	return "default"


def load_predictions_csv(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Load a predictions CSV produced by `experiments/validation.save_predictions`.

	Expected columns: index, y_true, y_pred
	"""
	_ensure_numpy()

	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Predictions CSV not found: {p}")

	with p.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		if reader.fieldnames is None:
			raise ValueError("Predictions CSV has no header row")

		required = {"index", "y_true", "y_pred"}
		if not required.issubset(set(reader.fieldnames)):
			raise ValueError(f"Predictions CSV must contain columns: {sorted(required)}")

		indices: list[int] = []
		y_true: list[Any] = []
		y_pred: list[Any] = []

		for row in reader:
			indices.append(int(row["index"]))
			y_true.append(row["y_true"])
			y_pred.append(row["y_pred"])

	return np.asarray(indices, dtype=int), np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)


def load_ucr_txt_split(
	data_dir: Union[str, Path],
	dataset_name: str,
	split: str,
) -> tuple[np.ndarray, np.ndarray]:
	"""Load a UCR split from a local .txt file.

	Expected path format:
	`<data_dir>/<dataset_name>/<dataset_name>_<split>.txt`
	"""
	_ensure_numpy()
	data_dir = Path(data_dir)
	split_up = split.upper()
	file_path = data_dir / dataset_name / f"{dataset_name}_{split_up}.txt"
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


def _categorical_to_numeric(
	labels: Sequence[Any],
) -> tuple[np.ndarray, list[Any]]:
	"""Map arbitrary labels to numeric indices for plotting."""
	# Preserve order of first appearance.
	unique: list[Any] = []
	mapping: dict[Any, int] = {}
	for v in labels:
		if v not in mapping:
			mapping[v] = len(unique)
			unique.append(v)
	return np.array([mapping[v] for v in labels], dtype=int), unique


def plot_overlay_by_correctness(
	dataset_name: str,
	predictions_csv: Union[str, Path],
	data_dir: Union[str, Path] = "data",
	split: str = "TEST",
	max_series: int | None = 60,
	save: bool = True,
	out_dir: Union[str, Path] = "visualization",
	figsize: Tuple[int, int] = (12, 6),
	color: str = "#0072B2",
	alpha: float = 0.15,
	seed: int = 0,
	include_labels: Sequence[Any] | None = None,
) -> plt:
	"""Overlay multiple series, highlighting incorrect predictions via transparency.

	Each class uses a distinct base color. Correct series are drawn with full opacity,
	while incorrect series are drawn with higher transparency so they are easy to spot.
	"""
	_ensure_matplotlib()
	_ensure_numpy()

	_, y_true, y_pred = load_predictions_csv(predictions_csv)
	X, _ = load_ucr_txt_split(data_dir=data_dir, dataset_name=dataset_name, split=split)

	if X.shape[0] == 0:
		raise ValueError("No time series available to plot")

	assert y_true.shape[0] == y_pred.shape[0]

	# Optionally filter to a subset of labels (only true labels are used for selection)
	if include_labels is not None:
		include_set = set(include_labels)
		mask = np.array([yt in include_set for yt in y_true], dtype=bool)
		X = X[mask]
		y_true = np.asarray(y_true)[mask]
		y_pred = np.asarray(y_pred)[mask]
		if X.shape[0] == 0:
			raise ValueError(f"No series matched include_labels={include_labels}")

	# Determine which series are correct / incorrect
	correct_mask = np.asarray(y_true) == np.asarray(y_pred)
	idx_correct = np.where(correct_mask)[0]
	idx_incorrect = np.where(~correct_mask)[0]

	# Select a subset for plotting (max_series=None or <=0 means "all")
	if max_series is None or max_series <= 0:
		n_correct = len(idx_correct)
		n_incorrect = len(idx_incorrect)
	else:
		n_correct = min(len(idx_correct), max_series)
		n_incorrect = min(len(idx_incorrect), max_series)

	rng = np.random.default_rng(seed)
	idx_correct = rng.choice(idx_correct, size=n_correct, replace=False) if n_correct > 0 else []
	idx_incorrect = rng.choice(idx_incorrect, size=n_incorrect, replace=False) if n_incorrect > 0 else []

	# Assign a distinct base color per true class.
	labels = list(dict.fromkeys(np.concatenate([y_true, y_pred])))
	colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"])
	label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(labels)}

	# Correct series should be fully visible, incorrect ones should be semi-transparent
	correct_alpha = 1.0
	incorrect_alpha = max(0.05, min(1.0, alpha))

	with plt.style.context("default"):
		fig, ax = plt.subplots(figsize=figsize)
		fig.patch.set_facecolor("white")
		ax.set_facecolor("white")

		for i in list(idx_correct) + list(idx_incorrect):
			label = y_true[i]
			is_correct = i in idx_correct
			color = label_to_color.get(label, color)
			alpha_plot = correct_alpha if is_correct else incorrect_alpha
			ax.plot(np.arange(X.shape[1]), X[i], color=color, alpha=alpha_plot)

		ax.set_title(
			f"{dataset_name} — overlay (transparent=incorrect) — {n_correct} correct / {n_incorrect} incorrect"
			+ (" (sampled)" if (len(idx_correct) < len(correct_mask) or len(idx_incorrect) < len(~correct_mask)) else "")
		)
		ax.set_xlabel("Time index")
		ax.set_ylabel("Value")
		ax.grid(True, alpha=0.18)

		# Legend handles: show correct and incorrect for each class
		from matplotlib.lines import Line2D

		handles = []
		for label, color_val in label_to_color.items():
			handles.append(Line2D([0], [0], color=color_val, lw=2, label=f"{label} (correct)", alpha=correct_alpha))
			handles.append(Line2D([0], [0], color=color_val, lw=2, label=f"{label} (incorrect)", alpha=incorrect_alpha))
		ax.legend(
			handles=handles,
			loc="upper left",
			framealpha=0.75,
			facecolor="white",
			edgecolor="black",
		)

		fig.tight_layout()

		if save:
			out_dir = _ensure_dir(out_dir)
			path = out_dir / f"{_safe_dataset_name(dataset_name)}_overlay_correctness.png"
			fig.savefig(path, dpi=200)

	return plt


def plot_confusion_matrix(
	y_true: Sequence[Any],
	y_pred: Sequence[Any],
	labels: Sequence[Any] | None = None,
	normalize: bool = False,
	save: bool = True,
	out_dir: Union[str, Path] = "visualization",
	dataset_name: Optional[str] = None,
	figsize: Tuple[int, int] = (6, 5),
) -> plt:
	"""Plot a confusion matrix for classification predictions."""
	_ensure_matplotlib()
	_ensure_numpy()

	y_true_arr = np.asarray(y_true).ravel()
	y_pred_arr = np.asarray(y_pred).ravel()

	if labels is None:
		labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
	else:
		labels = np.asarray(labels)

	label_to_idx = {l: i for i, l in enumerate(labels)}
	cm = np.zeros((len(labels), len(labels)), dtype=float)
	for t, p in zip(y_true_arr, y_pred_arr):
		if t in label_to_idx and p in label_to_idx:
			cm[label_to_idx[t], label_to_idx[p]] += 1

	if normalize:
		row_sums = cm.sum(axis=1, keepdims=True)
		row_sums[row_sums == 0] = 1
		cm = cm / row_sums

	title = "Confusion matrix"
	if dataset_name:
		title = f"{dataset_name} — {title}"

	with plt.style.context("default"):
		fig, ax = plt.subplots(figsize=figsize)
		fig.patch.set_facecolor("white")
		ax.set_facecolor("white")

		im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
		cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=-90, va="bottom")

		# Axis labels and ticks
		tick_labels = [str(v) for v in labels]
		ax.set_title(title, pad=20, fontsize=13, fontweight="bold")
		ax.set_xlabel("Predicted")
		ax.set_ylabel("True")
		ax.set_xticks(np.arange(len(labels)))
		ax.set_xticklabels(tick_labels, rotation=45, ha="right")
		ax.set_yticks(np.arange(len(labels)))
		ax.set_yticklabels(tick_labels)

		# Move X ticks to top for matrix-style layout
		ax.xaxis.set_ticks_position("top")
		ax.xaxis.set_label_position("top")

		# Add grid lines between cells
		ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
		ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
		ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
		ax.tick_params(which="minor", bottom=False, left=False)

		# Keep square cells
		ax.set_aspect("equal")

		for i in range(len(labels)):
			for j in range(len(labels)):
				val = cm[i, j]
				text = f"{val:.2f}" if normalize else f"{int(val)}"
				color = "white" if val > cm.max() / 2 else "black"
				ax.text(j, i, text, ha="center", va="center", color=color)

		fig.tight_layout()

		if save:
			out_dir = _ensure_dir(out_dir)
			fname = f"{_safe_dataset_name(dataset_name)}_confusion_matrix" + ("_norm" if normalize else "") + ".png"
			path = out_dir / fname
			fig.savefig(path, dpi=200)

	return plt


def _cli_main() -> None:
	"""Command-line entry point for prediction visualization."""
	import argparse

	parser = argparse.ArgumentParser(
		description="Visualize classifier predictions (from experiments/validation) on UCR datasets."
	)
	parser.add_argument(
		"--mode",
		choices=["overlay", "confusion"],
		default="overlay",
		help="Visualization mode (overlay/confusion).",
	)
	parser.add_argument(
		"--predictions",
		"-p",
		required=True,
		help="Path to predictions CSV (contains index,y_true,y_pred).",
	)
	parser.add_argument(
		"--dataset-name",
		"-d",
		required=True,
		help="UCR dataset name (used for loading split and naming output files).",
	)
	parser.add_argument(
		"--split",
		choices=["TRAIN", "TEST", "train", "test"],
		default="TEST",
		help="UCR split to load when plotting series/overlay.",
	)
	parser.add_argument(
		"--max-series",
		type=int,
		default=16,
		help="Maximum number of series to plot when mode=overlay (0 = all).",
	)
	parser.add_argument(
		"--include-labels",
		nargs="+",
		help="Optional list of true labels to include in the overlay plot (only those classes will be shown).",
	)
	parser.add_argument(
		"--normalize",
		action="store_true",
		help="Normalize confusion matrix values to percentages.",
	)
	parser.add_argument(
		"--out-dir",
		default="visualization",
		help="Directory where figures are saved.",
	)
	parser.add_argument(
		"--no-save",
		action="store_true",
		help="Do not save output files (useful for interactive sessions).",
	)

	args = parser.parse_args()

	if args.mode == "overlay":
		plot_overlay_by_correctness(
			dataset_name=args.dataset_name,
			predictions_csv=args.predictions,
			data_dir="data",
			split=args.split,
			max_series=args.max_series,
			save=not args.no_save,
			out_dir=args.out_dir,
			include_labels=args.include_labels,
		)
	else:  # confusion
		# confusion mode does not require dataset name because it only uses y_true/y_pred
		plot_confusion_matrix(
			*load_predictions_csv(args.predictions)[1:],
			labels=None,
			normalize=args.normalize,
			save=not args.no_save,
			out_dir=args.out_dir,
			dataset_name=args.dataset_name,
		)


if __name__ == "__main__":
	_cli_main()
