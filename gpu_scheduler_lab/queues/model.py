from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ResourceVector:
    gpu_units: float = 0.0
    gpu_memory_gb: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("gpu_units", self.gpu_units),
            ("gpu_memory_gb", self.gpu_memory_gb),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def __add__(self, other: ResourceVector) -> ResourceVector:
        return ResourceVector(
            self.gpu_units + other.gpu_units,
            self.gpu_memory_gb + other.gpu_memory_gb,
        )

    def __sub__(self, other: ResourceVector) -> ResourceVector:
        return ResourceVector(
            max(0.0, self.gpu_units - other.gpu_units),
            max(0.0, self.gpu_memory_gb - other.gpu_memory_gb),
        )

    def scale(self, factor: float) -> ResourceVector:
        if not math.isfinite(factor) or factor < 0:
            raise ValueError("resource scale must be finite and non-negative")
        return ResourceVector(self.gpu_units * factor, self.gpu_memory_gb * factor)

    def fits_within(self, limit: ResourceVector | None) -> bool:
        if limit is None:
            return True
        return (
            self.gpu_units <= limit.gpu_units + 1e-9
            and self.gpu_memory_gb <= limit.gpu_memory_gb + 1e-9
        )

    def to_dict(self) -> dict[str, float]:
        return {"gpu_units": self.gpu_units, "gpu_memory_gb": self.gpu_memory_gb}

    @classmethod
    def from_dict(cls, data: Any, *, default: ResourceVector | None = None) -> ResourceVector:
        if data is None:
            return default or cls()
        if not isinstance(data, dict):
            raise ValueError("resource vector must be a mapping")
        return cls(
            gpu_units=float(data.get("gpu_units", 0.0)),
            gpu_memory_gb=float(data.get("gpu_memory_gb", 0.0)),
        )

    @classmethod
    def limit_from_dict(cls, data: Any) -> ResourceVector | None:
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("resource limit must be a mapping")
        return cls(
            gpu_units=float(data.get("gpu_units", sys.float_info.max)),
            gpu_memory_gb=float(data.get("gpu_memory_gb", sys.float_info.max)),
        )


@dataclass(frozen=True, slots=True)
class QueueSpec:
    id: str
    parent: str | None
    weight: float = 1.0
    guaranteed: ResourceVector = field(default_factory=ResourceVector)
    limit: ResourceVector | None = None
    borrowing_enabled: bool = True
    reclaimable: bool = True
    priority_offset: int = 0

    def __post_init__(self) -> None:
        if not self.id or self.id.startswith("/") or self.id.endswith("/"):
            raise ValueError("queue id must be a non-empty normalized path")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("queue weight must be finite and positive")
        if self.limit is not None and not self.guaranteed.fits_within(self.limit):
            raise ValueError(f"queue {self.id} guarantee exceeds its limit")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueueSpec:
        queue_id = str(data.get("id", "")).strip()
        raw_parent = data.get("parent")
        return cls(
            id=queue_id,
            parent=str(raw_parent).strip() if raw_parent is not None else None,
            weight=float(data.get("weight", 1.0)),
            guaranteed=ResourceVector.from_dict(data.get("guaranteed")),
            limit=ResourceVector.limit_from_dict(data.get("limit")),
            borrowing_enabled=bool(data.get("borrowing_enabled", True)),
            reclaimable=bool(data.get("reclaimable", True)),
            priority_offset=int(data.get("priority_offset", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "parent": self.parent,
            "weight": self.weight,
            "guaranteed": self.guaranteed.to_dict(),
            "borrowing_enabled": self.borrowing_enabled,
            "reclaimable": self.reclaimable,
        }
        if self.limit is not None:
            payload["limit"] = self.limit.to_dict()
        if self.priority_offset:
            payload["priority_offset"] = self.priority_offset
        return payload
