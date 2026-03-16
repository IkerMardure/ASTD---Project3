"""Shared specification type for benchmark classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    """Specification needed to instantiate a benchmark estimator."""

    name: str
    module: str
    class_name: str
    kwargs: dict[str, Any]
