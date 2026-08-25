from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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

    def prepare(
        self,
        now: float,
        cluster: Cluster,
        pending: list[Job],
        running: list[Job],
    ) -> None:
        """Observe one scheduling pass without mutating simulation state."""
        return None

    def on_job_started(self, job: Job, now: float) -> None:
        """Receive a committed start after the engine owns the state change."""
        return None

    def metrics(self) -> dict[str, Any]:
        return {}
