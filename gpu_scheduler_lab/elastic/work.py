from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ElasticSpec:
    min_replicas: int
    preferred_replicas: int
    max_replicas: int
    scaling_efficiency: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.min_replicas <= self.preferred_replicas <= self.max_replicas:
            raise ValueError("elastic replicas must satisfy 0 < min <= preferred <= max")
        for replicas, efficiency in self.scaling_efficiency.items():
            if replicas <= 0 or not math.isfinite(efficiency) or efficiency <= 0:
                raise ValueError("scaling efficiency entries must be finite and positive")

    def efficiency(self, replicas: int) -> float:
        return self.scaling_efficiency.get(replicas, 1.0)

    def work_rate(self, replicas: int) -> float:
        if not self.min_replicas <= replicas <= self.max_replicas:
            raise ValueError("elastic replica count is outside configured bounds")
        return replicas * self.efficiency(replicas)

    @classmethod
    def from_dict(cls, data: Any) -> ElasticSpec | None:
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("elastic must be a mapping")
        raw_curve = data.get("scaling_efficiency", {})
        if not isinstance(raw_curve, dict):
            raise ValueError("elastic.scaling_efficiency must be a mapping")
        return cls(
            min_replicas=int(data["min_replicas"]),
            preferred_replicas=int(data["preferred_replicas"]),
            max_replicas=int(data["max_replicas"]),
            scaling_efficiency={int(k): float(v) for k, v in raw_curve.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "min_replicas": self.min_replicas,
            "preferred_replicas": self.preferred_replicas,
            "max_replicas": self.max_replicas,
        }
        if self.scaling_efficiency:
            payload["scaling_efficiency"] = dict(sorted(self.scaling_efficiency.items()))
        return payload
