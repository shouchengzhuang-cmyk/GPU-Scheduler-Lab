from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class NormalizationStats:
    source_rows: int = 0
    window_filtered_rows: int = 0
    sampled_rows: int = 0
    invalid_rows: int = 0
    selected_rows: int = 0
    jobs_without_gpu_model: int = 0
    time_origin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_sample(identifier: str, *, rate: float, seed: int) -> bool:
    if rate >= 1.0:
        return True
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < rate


def normalize_timestamps(values: Iterable[float]) -> tuple[list[float], float]:
    materialized = list(values)
    if not materialized:
        return [], 0.0
    origin = min(materialized)
    return [value - origin for value in materialized], origin
