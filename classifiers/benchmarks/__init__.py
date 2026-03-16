"""Benchmark classifiers package."""

from classifiers.benchmarks.suite import (
    DEFAULT_BENCHMARK_SPECS,
    instantiate_benchmark,
)

__all__ = ["DEFAULT_BENCHMARK_SPECS", "instantiate_benchmark"]
