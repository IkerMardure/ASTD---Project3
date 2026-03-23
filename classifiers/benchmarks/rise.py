"""Default RISE benchmark specification."""

from __future__ import annotations

from classifiers.benchmarks.spec import BenchmarkSpec


SPEC = BenchmarkSpec(
    name="RISE",
    module="aeon.classification.interval_based",
    class_name="RandomIntervalSpectralEnsembleClassifier",
    kwargs={},
)
