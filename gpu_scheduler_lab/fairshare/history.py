from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class DecayedUsageHistory:
    half_life: float = 300.0
    service: dict[str, float] = field(default_factory=dict)
    _last_time: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.half_life) or self.half_life <= 0:
            raise ValueError("fair-share half_life must be finite and positive")

    def integrate(self, now: float, rates: dict[str, float]) -> None:
        if not math.isfinite(now) or now < self._last_time:
            raise ValueError("historical usage time must be finite and monotonic")
        delta = now - self._last_time
        decay = math.exp(-math.log(2.0) * delta / self.half_life)
        keys = set(self.service) | set(rates)
        for queue_id in keys:
            rate = rates.get(queue_id, 0.0)
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("historical service rate must be finite and non-negative")
            self.service[queue_id] = self.service.get(queue_id, 0.0) * decay + rate * delta
        self._last_time = now

    def debt(self, queue_id: str, weights: dict[str, float]) -> float:
        normalized = {key: self.service.get(key, 0.0) / weight for key, weight in weights.items()}
        baseline = min(normalized.values(), default=0.0)
        result = normalized.get(queue_id, 0.0) - baseline
        if not math.isfinite(result):
            raise ValueError("fair-share debt must be finite")
        return result
