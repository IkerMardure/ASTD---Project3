"""Default InceptionTime benchmark specification."""

from __future__ import annotations

from classifiers.benchmarks.spec import BenchmarkSpec


SPEC = BenchmarkSpec(
    name="InceptionTime",
    module="aeon.classification.deep_learning",
    class_name="InceptionTimeClassifier",
    # Default is 1500 epochs which is impractical for benchmarking comparisons.
    # 50 epochs gives a fair evaluation at a reasonable cost.
    kwargs={"n_epochs": 50},
)
