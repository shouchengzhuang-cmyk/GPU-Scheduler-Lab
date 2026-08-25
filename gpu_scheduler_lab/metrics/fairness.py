from __future__ import annotations

from collections.abc import Iterable


def jains_fairness_index(values: Iterable[float]) -> float:
    samples = [max(0.0, float(value)) for value in values]
    if not samples:
        return 1.0
    denominator = len(samples) * sum(value * value for value in samples)
    if denominator == 0:
        return 1.0
    return sum(samples) ** 2 / denominator
