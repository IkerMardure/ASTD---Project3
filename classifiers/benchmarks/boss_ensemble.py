"""Default BOSS-ensemble benchmark specification."""

from __future__ import annotations

from classifiers.benchmarks.spec import BenchmarkSpec


SPEC = BenchmarkSpec(
    name="BOSS-ensemble",
    module="aeon.classification.dictionary_based",
    class_name="BOSSEnsemble",
    kwargs={},
)
