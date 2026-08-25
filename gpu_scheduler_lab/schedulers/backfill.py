from __future__ import annotations

import math
from dataclasses import dataclass

from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.schedulers.topology import TopologyAwareScheduler


@dataclass(frozen=True, slots=True)
class Reservation:
    job_id: str
    created_at: float
    estimated_start_time: float
    required_gpu_count: int
    gpu_model: str | None
    allowed_gpu_models: tuple[str, ...]


class ReservationBackfillScheduler(TopologyAwareScheduler):
    """Conservative EASY-style backfill with an oracle completion estimate."""

    name = "backfill"

    def __init__(self) -> None:
        self.reservation: Reservation | None = None
        self._head_job_id: str | None = None
        self._now = 0.0
        self._allowed_backfills: set[str] = set()
        self._reservation_count = 0
        self._successful_backfills = 0
        self._reserved_waiting_time = 0.0
        self._backfill_gpu_time = 0.0
        self._delay_violations = 0

    def prepare(
        self,
        now: float,
        cluster: Cluster,
        pending: list[Job],
        running: list[Job],
    ) -> None:
        self._now = now
        self._allowed_backfills.clear()
        ordered = sorted(pending, key=lambda job: (job.arrival_time, job.id))
        self._head_job_id = ordered[0].id if ordered else None
        if not ordered:
            self.reservation = None
            return
        head = ordered[0]
        if self.reservation is not None and self.reservation.job_id != head.id:
            self.reservation = None
        if super().place(cluster, head) is not None:
            return
        estimate = self._estimate_start(now, cluster, head, running)
        if estimate is None:
            if self.reservation is not None and self.reservation.job_id == head.id:
                self.reservation = None
            return
        if self.reservation is None or self.reservation.job_id != head.id:
            self.reservation = Reservation(
                job_id=head.id,
                created_at=now,
                estimated_start_time=estimate,
                required_gpu_count=head.gpu_count,
                gpu_model=head.gpu_model,
                allowed_gpu_models=head.allowed_gpu_models,
            )
            self._reservation_count += 1
        elif estimate < self.reservation.estimated_start_time:
            self.reservation = Reservation(
                job_id=head.id,
                created_at=self.reservation.created_at,
                estimated_start_time=estimate,
                required_gpu_count=head.gpu_count,
                gpu_model=head.gpu_model,
                allowed_gpu_models=head.allowed_gpu_models,
            )

    def pending_key(self, job: Job, now: float) -> tuple[float | int | str, ...]:
        if self._head_job_id == job.id:
            return (0, job.arrival_time, job.id)
        return (1, job.remaining_duration, job.arrival_time, job.id)

    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        placement = super().place(cluster, job)
        if placement is None:
            return None
        reservation = self.reservation
        if reservation is None or job.id == reservation.job_id:
            return placement
        completion = self._now + job.remaining_duration
        if job.preemption_count:
            completion += job.restart_cost
        if completion <= reservation.estimated_start_time:
            self._allowed_backfills.add(job.id)
            return placement
        return None

    def on_job_started(self, job: Job, now: float) -> None:
        reservation = self.reservation
        if reservation is not None and job.id == reservation.job_id:
            if now > reservation.estimated_start_time + 1e-9:
                self._delay_violations += 1
            self._reserved_waiting_time += now - reservation.created_at
            self.reservation = None
            return
        if job.id in self._allowed_backfills:
            self._successful_backfills += 1
            self._backfill_gpu_time += job.remaining_duration * job.gpu_count
            self._allowed_backfills.remove(job.id)

    def metrics(self) -> dict[str, float | int]:
        return {
            "reservation_count": self._reservation_count,
            "successful_backfill_count": self._successful_backfills,
            "reserved_job_waiting_time": self._reserved_waiting_time,
            "backfill_utilization_gain": self._backfill_gpu_time,
            "reservation_delay_violation_count": self._delay_violations,
        }

    def _estimate_start(
        self,
        now: float,
        cluster: Cluster,
        head: Job,
        running: list[Job],
    ) -> float | None:
        empty = cluster.clone(preserve_allocations=True)
        for gpu in empty.gpus:
            gpu.owner_job_id = None
            gpu.allocated_memory_gb = 0.0
        if super().place(empty, head) is None:
            return None

        projected = cluster.clone(preserve_allocations=True)
        completions = sorted(
            (
                _expected_completion(now, job),
                job.id,
            )
            for job in running
        )
        for completion, owner in completions:
            for gpu in projected.gpus:
                if gpu.owner_job_id == owner:
                    gpu.owner_job_id = None
                    gpu.allocated_memory_gb = 0.0
            if super().place(projected, head) is not None:
                return completion
        return None


def _expected_completion(now: float, job: Job) -> float:
    active = now - job.last_start_time if job.last_start_time is not None else 0.0
    remaining = max(0.0, job.duration - job.accumulated_runtime - active)
    result = now + remaining
    if not math.isfinite(result):
        raise ValueError(f"non-finite completion estimate for {job.id}")
    return result
