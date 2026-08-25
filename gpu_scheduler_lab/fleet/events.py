from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class FleetEventType(StrEnum):
    NODE_JOIN = "NODE_JOIN"
    NODE_DRAIN = "NODE_DRAIN"
    NODE_FAIL = "NODE_FAIL"
    NODE_RECOVER = "NODE_RECOVER"
    CAPACITY_REVOKE = "CAPACITY_REVOKE"
    CAPACITY_RETURN = "CAPACITY_RETURN"


@dataclass(frozen=True, slots=True)
class FleetEvent:
    time: float
    event_type: FleetEventType
    node_id: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError("fleet event time must be finite and non-negative")
        if not self.node_id:
            raise ValueError("fleet event node_id must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FleetEvent:
        return cls(
            time=float(data["time"]),
            event_type=FleetEventType(str(data["type"])),
            node_id=str(data["node_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"time": self.time, "type": self.event_type.value, "node_id": self.node_id}
