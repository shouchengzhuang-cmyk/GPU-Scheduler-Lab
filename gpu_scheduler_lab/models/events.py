from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    JOB_COMPLETE = "JOB_COMPLETE"
    JOB_CHECKPOINT_COMPLETE = "JOB_CHECKPOINT_COMPLETE"
    JOB_RESTART_COMPLETE = "JOB_RESTART_COMPLETE"
    JOB_ARRIVAL = "JOB_ARRIVAL"
    JOB_ADMIT = "JOB_ADMIT"
    JOB_REJECT = "JOB_REJECT"
    JOB_START = "JOB_START"
    JOB_PREEMPT = "JOB_PREEMPT"
    JOB_RESTART = "JOB_RESTART"
    JOB_RESUME = "JOB_RESUME"
    ELASTIC_SCALE_UP = "ELASTIC_SCALE_UP"
    ELASTIC_SCALE_DOWN = "ELASTIC_SCALE_DOWN"
    NODE_JOIN = "NODE_JOIN"
    NODE_DRAIN = "NODE_DRAIN"
    NODE_FAIL = "NODE_FAIL"
    NODE_RECOVER = "NODE_RECOVER"
    CAPACITY_REVOKE = "CAPACITY_REVOKE"
    CAPACITY_RETURN = "CAPACITY_RETURN"
    SCHEDULER_TICK = "SCHEDULER_TICK"


EVENT_ORDER = {
    EventType.JOB_COMPLETE: 0,
    EventType.JOB_CHECKPOINT_COMPLETE: 1,
    EventType.JOB_RESTART_COMPLETE: 2,
    EventType.NODE_JOIN: 3,
    EventType.NODE_RECOVER: 3,
    EventType.CAPACITY_RETURN: 3,
    EventType.NODE_DRAIN: 4,
    EventType.NODE_FAIL: 4,
    EventType.CAPACITY_REVOKE: 4,
    EventType.JOB_ARRIVAL: 5,
    EventType.SCHEDULER_TICK: 6,
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
