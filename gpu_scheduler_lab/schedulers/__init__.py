from __future__ import annotations

from typing import TYPE_CHECKING

from gpu_scheduler_lab.allocation.allocator import FairShareScheduler
from gpu_scheduler_lab.queues.hierarchy import QueueHierarchy
from gpu_scheduler_lab.schedulers.backfill import Reservation, ReservationBackfillScheduler
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.schedulers.binpack import BinPackScheduler
from gpu_scheduler_lab.schedulers.fifo import FIFOScheduler
from gpu_scheduler_lab.schedulers.preemptive import PreemptiveScheduler
from gpu_scheduler_lab.schedulers.spread import SpreadScheduler
from gpu_scheduler_lab.schedulers.topology import TopologyAwareScheduler

if TYPE_CHECKING:
    from gpu_scheduler_lab.scenario import Scenario


def create_scheduler(name: str, scenario: Scenario | None = None) -> Scheduler:
    normalized = name.lower()
    if normalized in {
        "drf",
        "historical-drf",
        "fairshare-no-borrow",
        "fairshare-borrow",
        "fairshare-reclaim",
    }:
        if scenario is None:
            raise ValueError(f"scheduler {name!r} requires a scenario")
        hierarchy = QueueHierarchy(scenario.queues)
        return FairShareScheduler(
            hierarchy,
            scenario.accounting,
            historical=normalized == "historical-drf",
            half_life=scenario.fairshare_half_life,
            borrowing=normalized != "fairshare-no-borrow",
            reclaim=normalized in {"fairshare-reclaim", "historical-drf"},
            name=normalized,
        )
    schedulers: dict[str, type[Scheduler]] = {
        "fifo": FIFOScheduler,
        "binpack": BinPackScheduler,
        "backfill": ReservationBackfillScheduler,
        "spread": SpreadScheduler,
        "preemptive": PreemptiveScheduler,
        "topology": TopologyAwareScheduler,
    }
    try:
        return schedulers[normalized]()
    except KeyError as exc:
        supported = ", ".join(sorted(schedulers))
        raise ValueError(f"unknown scheduler {name!r}; choose from: {supported}") from exc


__all__ = [
    "BinPackScheduler",
    "FIFOScheduler",
    "FairShareScheduler",
    "PreemptiveScheduler",
    "Reservation",
    "ReservationBackfillScheduler",
    "Scheduler",
    "SpreadScheduler",
    "TopologyAwareScheduler",
    "create_scheduler",
]
