from __future__ import annotations

import math

from gpu_scheduler_lab.queues.model import ResourceVector


def dominant_share(usage: ResourceVector, capacity: ResourceVector) -> float:
    shares = [
        usage.gpu_units / capacity.gpu_units if capacity.gpu_units else 0.0,
        usage.gpu_memory_gb / capacity.gpu_memory_gb if capacity.gpu_memory_gb else 0.0,
    ]
    result = max(shares)
    if not math.isfinite(result):
        raise ValueError("dominant share must be finite")
    return result


def weighted_dominant_share(
    usage: ResourceVector, capacity: ResourceVector, weight: float
) -> float:
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("DRF weight must be finite and positive")
    return dominant_share(usage, capacity) / weight
