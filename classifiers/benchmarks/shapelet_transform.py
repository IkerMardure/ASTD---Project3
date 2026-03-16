"""Default Shapelet Transform (ST) benchmark specification."""

from __future__ import annotations

from classifiers.benchmarks.spec import BenchmarkSpec


SPEC = BenchmarkSpec(
    name="Shapelet Transform (ST)",
    module="aeon.classification.shapelet_based",
    class_name="ShapeletTransformClassifier",
    kwargs={},
)
