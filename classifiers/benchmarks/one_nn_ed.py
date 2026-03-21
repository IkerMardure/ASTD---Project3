"""Default 1NN-ED benchmark specification.

ED stands for Euclidean Distance. This is typically a lightweight baseline
compared with deep models such as InceptionTime.
"""

from __future__ import annotations

from classifiers.benchmarks.spec import BenchmarkSpec


SPEC = BenchmarkSpec(
    name="1NN-ED",
    module="aeon.classification.distance_based",
    class_name="KNeighborsTimeSeriesClassifier",
    kwargs={"n_neighbors": 1, "distance": "euclidean"},
)
