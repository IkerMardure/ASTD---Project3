"""Default Rocket benchmark specification."""

from __future__ import annotations

from classifiers.benchmarks.spec import BenchmarkSpec


SPEC = BenchmarkSpec(
    name="Rocket",
    module="aeon.classification.convolution_based",
    class_name="RocketClassifier",
    kwargs={},
)
