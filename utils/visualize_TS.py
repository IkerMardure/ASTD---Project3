"""Utilities for visualizing UCR-style time series datasets.

The UCR dataset format (as used in this repo under `data/`) stores each sample as a
row where the first value is the class label and the remaining values are the
time-series values.

This module provides convenience functions to plot one time series or a small
subset of a dataset, and optionally save the resulting figures to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

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


def load_ucr_txt_dataset(
	file_path: Union[str, Path],
	# If the file contains a header row, set this to True and adjust `skiprows`.
	skiprows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
	"""Load a UCR-style text dataset.

	Parameters
	----------
	file_path:
		Path to a UCR .txt/.ts file.
	skiprows:
		Number of rows to skip at the beginning (e.g., header lines).

	Returns
	-------
	X: np.ndarray
		Shape (n_samples, n_timestamps) with series values.
	y: np.ndarray
		Shape (n_samples,) with class labels.
	"""

	_ensure_numpy()
	arr = np.loadtxt(file_path, skiprows=skiprows)
	if arr.ndim == 1:
		arr = arr.reshape(1, -1)

	y = arr[:, 0]
	X = arr[:, 1:]
	return X, y


def _get_preferred_style() -> str:
	"""Return an available matplotlib style for consistent plots."""
	candidates = ["seaborn-darkgrid", "seaborn", "ggplot", "tableau-colorblind10", "classic"]
	available = set(plt.style.available) if plt is not None else set()
	for name in candidates:
		if name in available:
			return name
	# Fallback to default if nothing matches
	return "default"


def generate_one_graph(
	series: np.ndarray,
	dataset_name: str,
	index: int,
	label: Optional[Any] = None,
	save: bool = False,
	out_dir: Union[str, Path] = "visualization",
	figsize: Tuple[int, int] = (10, 3.5),
) -> plt:
	"""Generate a line plot for a single time series.

	Parameters
	----------
	series:
		1D array of time series values.
	dataset_name:
		Name of the dataset (used for titles and filenames).
	index:
		Index of the series (used for filename and title).
	label:
		Optional class label. Included in the title and filename if provided.
	save:
		If True, saves the figure to `<out_dir>/<dataset_name>_<index>[_label<..>].png`.
	out_dir:
		Directory to save figures into.
	figsize:
		Figure size in inches.

	Returns
	-------
	matplotlib.pyplot
		The pyplot module (with the figure active) so callers can further customize.
	"""

	_ensure_matplotlib()
	series = np.asarray(series).ravel()
	out_dir = _ensure_dir(out_dir)

	title = f"{dataset_name} — series {index}"
	if label is not None:
		title += f" (label={label})"

	style = _get_preferred_style()
	with plt.style.context(style):
		fig, ax = plt.subplots(figsize=figsize)
		x = np.arange(series.shape[0])
		ax.plot(x, series, color="#0072B2", linewidth=1.75)
		ax.set_title(title, fontsize=13, fontweight="bold")
		ax.set_xlabel("Time index", fontsize=11)
		ax.set_ylabel("Value", fontsize=11)
		ax.grid(True, alpha=0.35)
		ax.margins(x=0)
		fig.tight_layout()

		if save:
			filename = f"{dataset_name}_{index}"
			if label is not None:
				filename += f"_label{label}"
			filename += ".png"
			path = out_dir / filename
			fig.savefig(path, dpi=200)

	return plt


def generate_dataset_graph(
	X: np.ndarray,
	dataset_name: str,
	labels: Optional[Sequence[Any]] = None,
	include_labels: Optional[Sequence[Any]] = None,
	max_series: int = 16,
	save: bool = True,
	out_dir: Union[str, Path] = "visualization",
	grid_shape: Tuple[int, int] = (4, 4),
	figsize: Tuple[int, int] = (14, 10),
) -> plt:
	"""Generate a grid of time series plots from a dataset.

	This is useful for quickly inspecting shape/quality of the first few samples.

	Parameters
	----------
	X:
		Array with shape (n_samples, n_timestamps).
	dataset_name:
		Name used for titles and output filename.
	labels:
		Optional array of labels with length n_samples.
	include_labels:
		Optional subset of labels to include. If provided, only samples whose
		labels are in this list are plotted.
	max_series:
		Maximum number of series to plot (will take the first `max_series`).
	save:
		If True, saves the figure to `<out_dir>/<dataset_name>.png`.
	out_dir:
		Directory to save figures into.
	grid_shape:
		Number of rows/cols for the plot grid (rows, cols).
	figsize:
		Figure size in inches.

	Returns
	-------
	matplotlib.pyplot
		The pyplot module (with the figure active) so callers can further customize.
	"""

	_ensure_matplotlib()
	X = np.asarray(X)
	if X.ndim != 2:
		raise ValueError("X must be a 2D array of shape (n_samples, n_timestamps)")

	labels_arr = np.asarray(labels) if labels is not None else None
	if include_labels is not None:
		if labels_arr is None:
			raise ValueError("include_labels requires labels to be provided")
		include_set = set(include_labels)
		mask = np.array([lab in include_set for lab in labels_arr], dtype=bool)
		X = X[mask]
		labels_arr = labels_arr[mask]
		if X.shape[0] == 0:
			raise ValueError(f"No series matched include_labels={include_labels}")

	n_samples = min(X.shape[0], max_series)
	rows, cols = grid_shape
	if rows * cols < 1:
		raise ValueError("grid_shape must contain at least one subplot")

	# Adjust grid to fit the requested number of plots.
	n_plots = min(n_samples, rows * cols)
	rows = int(np.ceil(n_plots / cols))

	out_dir = _ensure_dir(out_dir)

	style = _get_preferred_style()
	with plt.style.context(style):
		fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
		axes_flat = np.array(axes).reshape(-1)

		for i in range(rows * cols):
			ax = axes_flat[i]
			if i >= n_plots:
				ax.axis("off")
				continue

			series = X[i]
			x = np.arange(series.shape[0])
			ax.plot(x, series, color="#0072B2", linewidth=1.25)

			title = f"#{i}"
			if labels_arr is not None:
				try:
					label = labels_arr[i]
					title += f" (label={label})"
				except Exception:
					# If labels has a different length or is not indexable, ignore.
					pass
			ax.set_title(title, fontsize=10)
			ax.grid(True, alpha=0.25)

			if i % cols == 0:
				ax.set_ylabel("Value", fontsize=9)
			if i >= cols * (rows - 1):
				ax.set_xlabel("Time index", fontsize=9)

		fig.suptitle(f"{dataset_name} (first {n_plots} samples)", fontsize=14, fontweight="bold")
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])

		if save:
			path = out_dir / f"{dataset_name}.png"
			fig.savefig(path, dpi=200)

	# Also provide an overlay plot showing all series on one axes.
	# This makes it easier to inspect overall shape/variance across samples.
	style = _get_preferred_style()
	with plt.style.context(style):
		fig2, ax2 = plt.subplots(figsize=(12, 5))
		if labels_arr is not None:
			unique_labels = list(dict.fromkeys(labels_arr[:n_plots]))
			palette = plt.rcParams.get("axes.prop_cycle").by_key().get(
				"color", ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
			)
			label_to_color = {
				label: palette[idx % len(palette)] for idx, label in enumerate(unique_labels)
			}

			# Keep class '1' always blue, even when filtering to only that label.
			for label in unique_labels:
				label_str = str(label).strip()
				if label_str in {"1", "1.0"}:
					label_to_color[label] = "#0072B2"
		else:
			label_to_color = {}

		for i in range(n_plots):
			series = X[i]
			x = np.arange(series.shape[0])
			line_color = "#0072B2"
			if labels_arr is not None:
				line_color = label_to_color.get(labels_arr[i], line_color)
			ax2.plot(x, series, color=line_color, alpha=0.35, linewidth=1.0)
		ax2.set_title(f"{dataset_name} (overlay of first {n_plots} series)", fontsize=13, fontweight="bold")
		ax2.set_xlabel("Time index", fontsize=11)
		ax2.set_ylabel("Value", fontsize=11)
		ax2.grid(True, alpha=0.25)
		ax2.margins(x=0)
		if labels_arr is not None and len(label_to_color) > 0:
			from matplotlib.lines import Line2D
			handles = [
				Line2D([0], [0], color=col, lw=2, label=str(lab))
				for lab, col in label_to_color.items()
			]
			ax2.legend(handles=handles, title="Label", loc="upper left", framealpha=0.75)
		fig2.tight_layout()

		if save:
			path2 = out_dir / f"{dataset_name}_overlay.png"
			fig2.savefig(path2, dpi=200)

	return plt


def _cli_main() -> None:
	"""Command-line entry point for quick plotting."""
	import argparse

	parser = argparse.ArgumentParser(
		description="Generate and save simple visualizations for UCR-style time series datasets."
	)
	parser.add_argument("--input", "-i", required=True, help="Path to a UCR .txt/.ts dataset file.")
	parser.add_argument(
		"--dataset-name",
		"-d",
		help="Name of the dataset (used in output filenames). If omitted, derived from input filename.",
	)
	parser.add_argument(
		"--mode",
		choices=["one", "grid"],
		default="grid",
		help="Plot a single series (one) or a grid of series (grid).",
	)
	parser.add_argument(
		"--index",
		type=int,
		default=0,
		help="Index of the series to plot when mode=one.",
	)
	parser.add_argument(
		"--max-series",
		type=int,
		default=16,
		help="Maximum number of series to plot when mode=grid.",
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

	# Load data
	X, y = load_ucr_txt_dataset(args.input)
	dataset_name = args.dataset_name or Path(args.input).stem

	if args.mode == "one":
		generate_one_graph(
			X[args.index],
			dataset_name=dataset_name,
			index=args.index,
			label=y[args.index] if len(y) > args.index else None,
			save=not args.no_save,
			out_dir=args.out_dir,
		)
	else:
		generate_dataset_graph(
			X,
			dataset_name=dataset_name,
			labels=y,
			max_series=args.max_series,
			save=not args.no_save,
			out_dir=args.out_dir,
		)


if __name__ == "__main__":
	_cli_main()
