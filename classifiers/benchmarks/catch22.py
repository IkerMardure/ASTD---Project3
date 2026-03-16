"""Default catch22 benchmark specification."""

from __future__ import annotations

from classifiers.benchmarks.spec import BenchmarkSpec


SPEC = BenchmarkSpec(
    name="catch22",
    module="aeon.classification.feature_based",
    class_name="Catch22Classifier",
    kwargs={},
)
