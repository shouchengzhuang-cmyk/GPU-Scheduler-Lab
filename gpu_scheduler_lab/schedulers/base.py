from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job


class Scheduler(ABC):
    name = "base"
    supports_preemption = False
    supports_reclaim = False
    supports_elastic = False
    supports_guarantee_placement = False
    dynamic_pending_order = False
    aging_interval = 30.0

    @abstractmethod
    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        """Return a complete placement or None without mutating cluster state."""

    def pending_key(self, job: Job, now: float) -> tuple[Any, ...]:
        return (job.arrival_time, job.id)

    def place_guaranteed(self, cluster: Cluster, job: Job) -> list[str] | None:
        """Return a placement only when it consumes remaining guaranteed entitlement."""
        return None

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

    def can_reclaim(self, victim: Job, incoming: Job) -> bool:
        return False

    def can_reclaim_request(self, incoming: Job) -> bool:
        return False

    def can_reclaim_placement(
        self,
        incoming: Job,
        gpu_ids: list[str],
        entitlement_queue_ids: set[str] | None = None,
    ) -> bool:
        return False

    def reclaim_entitlement_queue(self, victim: Job, incoming: Job) -> str | None:
        return None

    def can_reclaim_allocation(
        self,
        victim: Job,
        incoming: Job,
        released_gpu_ids: list[str],
        *,
        allow_indivisible_collateral: bool = False,
    ) -> bool:
        return False

    def can_scale_up(self, job: Job) -> bool:
        return False

    def scale_up_key(self, job: Job, now: float) -> tuple[Any, ...]:
        return self.pending_key(job, now)

    def can_resize(self, job: Job, replicas: int) -> bool:
        return False

    def can_resize_placement(self, job: Job, gpu_ids: list[str]) -> bool:
        """Validate a concrete resize placement after GPU selection."""
        return True
