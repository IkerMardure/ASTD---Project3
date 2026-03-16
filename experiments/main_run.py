"""Minimal experiment runner for Aeon Time Series Forest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from classifiers.tsf_classifier import AeonTSFClassifier, TSFConfig


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


def main() -> None:
	parser = argparse.ArgumentParser(description="Run TSF classifier on synthetic data.")
	parser.add_argument("--n-train", type=int, default=60)
	parser.add_argument("--n-test", type=int, default=20)
	parser.add_argument("--n-timestamps", type=int, default=80)
	parser.add_argument("--n-estimators", type=int, default=200)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

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


if __name__ == "__main__":
	main()
