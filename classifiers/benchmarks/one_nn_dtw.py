"""Default 1NN-DTW benchmark specification."""

from __future__ import annotations

from classifiers.benchmarks.spec import BenchmarkSpec


SPEC = BenchmarkSpec(
    name="1NN-DTW",
    module="aeon.classification.distance_based",
    class_name="KNeighborsTimeSeriesClassifier",
    kwargs={"n_neighbors": 1, "distance": "dtw", "n_jobs": -1},
)
