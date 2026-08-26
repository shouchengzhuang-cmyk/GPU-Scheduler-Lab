from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    JOB_COMPLETE = "JOB_COMPLETE"
    JOB_ARRIVAL = "JOB_ARRIVAL"
    JOB_START = "JOB_START"
    JOB_PREEMPT = "JOB_PREEMPT"
    JOB_RESUME = "JOB_RESUME"
    SCHEDULER_TICK = "SCHEDULER_TICK"


EVENT_ORDER = {
    EventType.JOB_COMPLETE: 0,
    EventType.JOB_ARRIVAL: 1,
    EventType.SCHEDULER_TICK: 2,
}


@dataclass(order=True, slots=True)
class Event:
    time: float
    order: int
    sequence: int
    event_type: EventType
    job_id: str
    generation: int = 0


@dataclass(frozen=True, slots=True)
class TraceRecord:
    time: float
    event: EventType
    job_id: str
    gpu_ids: tuple[str, ...] = ()
    node_ids: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event"] = self.event.value
        data["gpu_ids"] = list(self.gpu_ids)
        data["node_ids"] = list(self.node_ids)
        return data
