from gpu_scheduler_lab.schedulers.backfill import Reservation, ReservationBackfillScheduler
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.schedulers.binpack import BinPackScheduler
from gpu_scheduler_lab.schedulers.fifo import FIFOScheduler
from gpu_scheduler_lab.schedulers.preemptive import PreemptiveScheduler
from gpu_scheduler_lab.schedulers.spread import SpreadScheduler
from gpu_scheduler_lab.schedulers.topology import TopologyAwareScheduler


def create_scheduler(name: str) -> Scheduler:
    schedulers: dict[str, type[Scheduler]] = {
        "fifo": FIFOScheduler,
        "binpack": BinPackScheduler,
        "backfill": ReservationBackfillScheduler,
        "spread": SpreadScheduler,
        "preemptive": PreemptiveScheduler,
        "topology": TopologyAwareScheduler,
    }
    try:
        return schedulers[name.lower()]()
    except KeyError as exc:
        supported = ", ".join(sorted(schedulers))
        raise ValueError(f"unknown scheduler {name!r}; choose from: {supported}") from exc


__all__ = [
    "BinPackScheduler",
    "FIFOScheduler",
    "PreemptiveScheduler",
    "Reservation",
    "ReservationBackfillScheduler",
    "Scheduler",
    "SpreadScheduler",
    "TopologyAwareScheduler",
    "create_scheduler",
]
