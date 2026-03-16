"""Time Series Forest classifier wrapper using aeon.

This module provides a small, project-friendly API around aeon's
TimeSeriesForestClassifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np


def _to_3d_numpy(X: np.ndarray) -> np.ndarray:
	"""Convert input series to aeon expected shape: (n_cases, n_channels, n_timepoints)."""
	arr = np.asarray(X)

	if arr.ndim == 2:
		# Treat 2D data as univariate (n_cases, n_timepoints).
		arr = arr[:, np.newaxis, :]
	elif arr.ndim != 3:
		raise ValueError(
			"X must be a 2D or 3D array-like object. "
			f"Received shape {arr.shape} with {arr.ndim} dimensions."
		)

	if arr.shape[0] == 0:
		raise ValueError("X is empty. Provide at least one training instance.")

	return arr


@dataclass
class TSFConfig:
	"""Configuration for the aeon TimeSeriesForestClassifier."""

	n_estimators: int = 200
	min_interval: int = 3
	n_jobs: int = -1
	random_state: int | None = 42


class AeonTSFClassifier:
	"""Project wrapper for aeon's Time Series Forest classifier."""

	def __init__(self, config: TSFConfig | None = None, **kwargs: Any) -> None:
		try:
			tsf_cls = getattr(
				importlib.import_module("aeon.classification.interval_based"),
				"TimeSeriesForestClassifier",
			)
		except Exception as exc:  # pragma: no cover - informative import guard
			raise ImportError(
				"aeon is required for TSF. Install it with: pip install aeon"
			) from exc

		self.config = config or TSFConfig()
		params = {
			"n_estimators": self.config.n_estimators,
			"min_interval": self.config.min_interval,
			"n_jobs": self.config.n_jobs,
			"random_state": self.config.random_state,
		}
		params.update(kwargs)

		self.model = tsf_cls(**params)
		self._is_fitted = False

	def fit(self, X: np.ndarray, y: np.ndarray) -> "AeonTSFClassifier":
		"""Train the TSF model."""
		X_3d = _to_3d_numpy(X)
		y_arr = np.asarray(y)

		if X_3d.shape[0] != y_arr.shape[0]:
			raise ValueError(
				"X and y must contain the same number of samples. "
				f"Got {X_3d.shape[0]} and {y_arr.shape[0]}."
			)

		self.model.fit(X_3d, y_arr)
		self._is_fitted = True
		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Predict class labels."""
		self._check_is_fitted()
		X_3d = _to_3d_numpy(X)
		return self.model.predict(X_3d)

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		"""Predict class probabilities."""
		self._check_is_fitted()
		X_3d = _to_3d_numpy(X)
		return self.model.predict_proba(X_3d)

	def score(self, X: np.ndarray, y: np.ndarray) -> float:
		"""Compute classification accuracy."""
		y_true = np.asarray(y)
		y_pred = self.predict(X)
		if y_true.shape[0] != y_pred.shape[0]:
			raise ValueError(
				"y and predictions must contain the same number of samples. "
				f"Got {y_true.shape[0]} and {y_pred.shape[0]}."
			)
		return float(np.mean(y_true == y_pred))

	def _check_is_fitted(self) -> None:
		if not self._is_fitted:
			raise RuntimeError("Model is not fitted. Call fit before prediction.")


if __name__ == "__main__":
	# Quick smoke test with synthetic univariate time series.
	rng = np.random.default_rng(42)
	n_train, n_test, n_timestamps = 60, 20, 80

	X_train = rng.normal(size=(n_train, n_timestamps))
	X_test = rng.normal(size=(n_test, n_timestamps))

	# Label by mean sign to create an easy synthetic binary task.
	y_train = (X_train.mean(axis=1) > 0.0).astype(int)
	y_test = (X_test.mean(axis=1) > 0.0).astype(int)

	clf = AeonTSFClassifier()
	clf.fit(X_train, y_train)
	acc = clf.score(X_test, y_test)
	print(f"Synthetic test accuracy: {acc:.3f}")
