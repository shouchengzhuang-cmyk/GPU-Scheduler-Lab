from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_scheduler_lab.models.cluster import Cluster


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


_CAPACITY_GAIN_EVENTS = frozenset(
    {
        FleetEventType.NODE_JOIN,
        FleetEventType.NODE_RECOVER,
        FleetEventType.CAPACITY_RETURN,
    }
)


def schedulable_node_snapshots(
    cluster: Cluster,
    fleet_events: Iterable[FleetEvent],
    *,
    after: float | None = None,
) -> tuple[frozenset[str], ...]:
    """Return schedulable nodes that can coexist at each fleet timeline state."""
    gpu_node_ids = {node.id for node in cluster.nodes if node.gpus}
    active = {
        node.id
        for node in cluster.nodes
        if node.id in gpu_node_ids and node.available and node.schedulable
    }
    future_events = sorted(
        (event for event in fleet_events if after is None or event.time > after),
        key=lambda event: (
            event.time,
            0 if event.event_type in _CAPACITY_GAIN_EVENTS else 1,
            event.event_type.value,
            event.node_id,
        ),
    )
    include_initial = after is not None or not future_events or future_events[0].time > 0
    snapshots = [frozenset(active)] if include_initial else []
    for _, events_at_time in groupby(future_events, key=lambda event: event.time):
        for event in events_at_time:
            if event.node_id not in gpu_node_ids:
                continue
            if event.event_type in _CAPACITY_GAIN_EVENTS:
                active.add(event.node_id)
            else:
                active.discard(event.node_id)
        snapshot = frozenset(active)
        if not snapshots or snapshot != snapshots[-1]:
            snapshots.append(snapshot)
    return tuple(snapshots)
