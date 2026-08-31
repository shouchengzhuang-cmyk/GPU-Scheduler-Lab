from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any


class EvidenceKind(StrEnum):
    MEASURED = "MEASURED"
    ASSUMED = "ASSUMED"
    SYNTHETIC = "SYNTHETIC"


@dataclass(frozen=True, slots=True)
class PerformanceProfile:
    source_kind: EvidenceKind
    source_id: str
    model_variant: str
    ttft_ms: float
    tpot_ms: float
    throughput_tokens_s: float
    power_watts: float
    cost_per_hour: float

    def __post_init__(self) -> None:
        if not self.source_id or not self.model_variant:
            raise ValueError("performance profile source_id and model_variant must not be empty")
        positive = ("ttft_ms", "tpot_ms", "throughput_tokens_s")
        non_negative = ("power_watts", "cost_per_hour")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"performance profile {name} must be finite and positive")
        for name in non_negative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"performance profile {name} must be finite and non-negative")

    @classmethod
    def from_dict(cls, data: Any) -> PerformanceProfile:
        if not isinstance(data, dict):
            raise ValueError("performance profile must be a mapping")
        required = {field.name for field in fields(cls)}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"performance profile missing fields: {', '.join(missing)}")
        return cls(
            source_kind=EvidenceKind(str(data["source_kind"])),
            source_id=str(data["source_id"]),
            model_variant=str(data["model_variant"]),
            ttft_ms=float(data["ttft_ms"]),
            tpot_ms=float(data["tpot_ms"]),
            throughput_tokens_s=float(data["throughput_tokens_s"]),
            power_watts=float(data["power_watts"]),
            cost_per_hour=float(data["cost_per_hour"]),
        )

    def to_dict(self) -> dict[str, str | float]:
        return {
            "source_kind": self.source_kind.value,
            "source_id": self.source_id,
            "model_variant": self.model_variant,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "throughput_tokens_s": self.throughput_tokens_s,
            "power_watts": self.power_watts,
            "cost_per_hour": self.cost_per_hour,
        }
