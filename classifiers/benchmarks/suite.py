"""Benchmark suite definitions and model factories.

This module centralizes benchmark algorithm specifications so experiment
orchestration can stay focused on evaluation logic.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from classifiers.benchmarks.boss_ensemble import SPEC as BOSS_ENSEMBLE_SPEC
from classifiers.benchmarks.catch22 import SPEC as CATCH22_SPEC
from classifiers.benchmarks.inception_time import SPEC as INCEPTION_TIME_SPEC
from classifiers.benchmarks.one_nn_dtw import SPEC as ONE_NN_DTW_SPEC
from classifiers.benchmarks.rocket import SPEC as ROCKET_SPEC
from classifiers.benchmarks.shapelet_transform import SPEC as SHAPELET_TRANSFORM_SPEC
from classifiers.benchmarks.spec import BenchmarkSpec


DEFAULT_BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    ONE_NN_DTW_SPEC,
    BOSS_ENSEMBLE_SPEC,
    SHAPELET_TRANSFORM_SPEC,
    ROCKET_SPEC,
    INCEPTION_TIME_SPEC,
    CATCH22_SPEC,
)


def supports_parameter(cls: type[Any], param_name: str) -> bool:
    """Check if a constructor accepts a given parameter name."""
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    return param_name in signature.parameters


def instantiate_benchmark(spec: BenchmarkSpec, random_state: int | None) -> Any:
    """Instantiate one benchmark model with optional random_state injection."""
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.class_name)

    kwargs = dict(spec.kwargs)
    if (
        random_state is not None
        and "random_state" not in kwargs
        and supports_parameter(cls, "random_state")
    ):
        kwargs["random_state"] = random_state

    return cls(**kwargs)
