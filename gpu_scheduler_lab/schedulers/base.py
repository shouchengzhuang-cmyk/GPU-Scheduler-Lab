from __future__ import annotations

from abc import ABC, abstractmethod

from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job


class Scheduler(ABC):
    name = "base"
    supports_preemption = False
    aging_interval = 30.0

    @abstractmethod
    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        """Return a complete placement or None without mutating cluster state."""

    def pending_key(self, job: Job, now: float) -> tuple[float | int | str, ...]:
        return (job.arrival_time, job.id)
