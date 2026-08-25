from __future__ import annotations

from typing import Any

from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.fairshare.drf import weighted_dominant_share
from gpu_scheduler_lab.fairshare.history import DecayedUsageHistory
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.queues.hierarchy import QueueHierarchy
from gpu_scheduler_lab.queues.model import ResourceVector
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.schedulers.topology import TopologyAwareScheduler


class FairShareScheduler(Scheduler):
    """Queue allocation policy composed with an independent placement scheduler."""

    def __init__(
        self,
        hierarchy: QueueHierarchy,
        accounting: AccountingPolicy,
        *,
        placement: Scheduler | None = None,
        historical: bool = False,
        half_life: float = 300.0,
        borrowing: bool = True,
        reclaim: bool = False,
        elastic: bool = True,
        name: str | None = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.accounting = accounting
        self.placement = placement or TopologyAwareScheduler()
        self.historical = historical
        self.borrowing = borrowing
        self.supports_reclaim = reclaim
        self.supports_elastic = elastic
        self.name = name or ("historical-drf" if historical else "drf")
        self.history = DecayedUsageHistory(half_life)
        self._cluster: Cluster | None = None
        self._direct_usage: dict[str, ResourceVector] = {}
        self._aggregate_usage: dict[str, ResourceVector] = {}
        self._last_rates: dict[str, float] = {}
        self._debts: dict[str, float] = {}
        self._capacity = ResourceVector()

    def prepare(
        self,
        now: float,
        cluster: Cluster,
        pending: list[Job],
        running: list[Job],
    ) -> None:
        self.history.integrate(now, self._last_rates)
        self.refresh_usage(cluster, running)
        self._last_rates = {
            queue_id: usage.gpu_units for queue_id, usage in self._aggregate_usage.items()
        }
        self._debts = {
            queue_id: self.history.debt(
                queue_id,
                {
                    sibling_id: self.hierarchy.specs[sibling_id].weight
                    for sibling_id in self.hierarchy.children[spec.parent or "root"]
                },
            )
            for queue_id, spec in self.hierarchy.specs.items()
            if queue_id != "root"
        }
        self.placement.prepare(now, cluster, pending, running)

    def refresh_usage(self, cluster: Cluster, allocated: list[Job]) -> None:
        self._cluster = cluster
        self._capacity = self.accounting.capacity(cluster)
        direct: dict[str, ResourceVector] = {}
        for job in allocated:
            if not job.allocated_gpu_ids:
                continue
            direct[job.queue_id] = direct.get(
                job.queue_id, ResourceVector()
            ) + self.accounting.allocation(job, cluster)
        self._direct_usage = direct
        self._aggregate_usage = self.hierarchy.aggregate_usage(direct)
        for job in allocated:
            job.borrowed_gpu_units = 0.0
        by_queue: dict[str, list[Job]] = {}
        for job in allocated:
            if job.allocated_gpu_ids:
                by_queue.setdefault(job.queue_id, []).append(job)
        for queue_id, queue_jobs in by_queue.items():
            guarantee = self.hierarchy.specs[queue_id].guaranteed.gpu_units
            borrowed = max(0.0, direct[queue_id].gpu_units - guarantee)
            for job in sorted(queue_jobs, key=lambda item: (int(item.priority), item.id)):
                allocation = self.accounting.allocation(job, cluster).gpu_units
                job.borrowed_gpu_units = min(allocation, borrowed)
                borrowed -= job.borrowed_gpu_units
                if borrowed <= 0:
                    break

    def pending_key(self, job: Job, now: float) -> tuple[float | int | str, ...]:
        spec = self.hierarchy.specs[job.queue_id]
        usage = self._aggregate_usage.get(job.queue_id, ResourceVector())
        capacity = self._capacity
        guarantee_deficit = (
            usage.gpu_units + 1e-9 < spec.guaranteed.gpu_units
            or usage.gpu_memory_gb + 1e-9 < spec.guaranteed.gpu_memory_gb
        )
        dominant = weighted_dominant_share(usage, capacity, spec.weight)
        debt = self._debts.get(job.queue_id, 0.0) if self.historical else 0.0
        return (
            0 if guarantee_deficit else 1,
            debt,
            dominant,
            -(int(job.priority) + spec.priority_offset),
            job.arrival_time,
            job.id,
        )

    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        demand = self.accounting.demand(job, cluster)
        if not self.hierarchy.can_allocate(
            job.queue_id,
            demand,
            self._direct_usage,
            borrowing=self.borrowing,
            aggregate_usage=self._aggregate_usage,
        ):
            return None
        placement = self.placement.place(cluster, job)
        if placement is None:
            return None
        actual_demand = self.accounting.allocation(job, cluster, placement)
        if not self.hierarchy.can_allocate(
            job.queue_id,
            actual_demand,
            self._direct_usage,
            borrowing=self.borrowing,
            aggregate_usage=self._aggregate_usage,
        ):
            return None
        return placement

    def on_job_started(self, job: Job, now: float) -> None:
        if self._cluster is None:
            return
        before = self._aggregate_usage.get(job.queue_id, ResourceVector())
        allocation = self.accounting.allocation(job, self._cluster)
        guarantee = self.hierarchy.specs[job.queue_id].guaranteed.gpu_units
        remaining_entitlement = max(0.0, guarantee - before.gpu_units)
        job.borrowed_gpu_units = max(0.0, allocation.gpu_units - remaining_entitlement)
        self._direct_usage[job.queue_id] = (
            self._direct_usage.get(job.queue_id, ResourceVector()) + allocation
        )
        self._aggregate_usage = self.hierarchy.aggregate_usage(self._direct_usage)
        self.placement.on_job_started(job, now)

    def can_reclaim(self, victim: Job, incoming: Job) -> bool:
        if (
            victim.queue_id == incoming.queue_id
            or victim.borrowed_gpu_units <= 0
            or not self._usage_exceeds_guarantee(victim.queue_id)
        ):
            return False
        return all(
            self.hierarchy.specs[queue_id].reclaimable
            for queue_id in self.hierarchy.ancestors(victim.queue_id)
        ) and self._has_guarantee_deficit(incoming.queue_id)

    def can_scale_up(self, job: Job) -> bool:
        if not self.supports_elastic or job.elastic is None:
            return False
        return self._has_guarantee_deficit(job.queue_id)

    def _has_guarantee_deficit(self, queue_id: str) -> bool:
        for ancestor_id in self.hierarchy.ancestors(queue_id):
            spec = self.hierarchy.specs[ancestor_id]
            dimensions = spec.guaranteed_dimensions or frozenset()
            usage = self._aggregate_usage.get(ancestor_id, ResourceVector())
            if (
                "gpu_units" in dimensions and usage.gpu_units + 1e-9 < spec.guaranteed.gpu_units
            ) or (
                "gpu_memory_gb" in dimensions
                and usage.gpu_memory_gb + 1e-9 < spec.guaranteed.gpu_memory_gb
            ):
                return True
        return False

    def _usage_exceeds_guarantee(self, queue_id: str) -> bool:
        for ancestor_id in self.hierarchy.ancestors(queue_id):
            spec = self.hierarchy.specs[ancestor_id]
            dimensions = spec.guaranteed_dimensions or frozenset()
            usage = self._aggregate_usage.get(ancestor_id, ResourceVector())
            if (
                "gpu_units" in dimensions and usage.gpu_units > spec.guaranteed.gpu_units + 1e-9
            ) or (
                "gpu_memory_gb" in dimensions
                and usage.gpu_memory_gb > spec.guaranteed.gpu_memory_gb + 1e-9
            ):
                return True
        return not any(
            self.hierarchy.specs[ancestor_id].guaranteed_dimensions
            for ancestor_id in self.hierarchy.ancestors(queue_id)
        )

    def can_resize(self, job: Job, replicas: int) -> bool:
        if self._cluster is None or replicas <= job.current_replicas:
            return False
        additional = replicas - job.current_replicas
        demand = self.accounting.demand(job, self._cluster, additional)
        return self.hierarchy.can_allocate(
            job.queue_id,
            demand,
            self._direct_usage,
            borrowing=self.borrowing,
            aggregate_usage=self._aggregate_usage,
        )

    def can_resize_placement(self, job: Job, gpu_ids: list[str]) -> bool:
        if self._cluster is None:
            return False
        current = set(job.allocated_gpu_ids)
        added = [gpu_id for gpu_id in gpu_ids if gpu_id not in current]
        demand = self.accounting.allocation(job, self._cluster, added)
        return self.hierarchy.can_allocate(
            job.queue_id,
            demand,
            self._direct_usage,
            borrowing=self.borrowing,
            aggregate_usage=self._aggregate_usage,
        )

    def queue_snapshot(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for queue_id, spec in sorted(self.hierarchy.specs.items()):
            usage = self._aggregate_usage.get(queue_id, ResourceVector())
            guaranteed = spec.guaranteed.gpu_units
            siblings = self.hierarchy.children[spec.parent or "root"]
            sibling_weight = sum(self.hierarchy.specs[item].weight for item in siblings)
            result[queue_id] = {
                "gpu_units": usage.gpu_units,
                "gpu_memory_gb": usage.gpu_memory_gb,
                "guaranteed_usage": min(usage.gpu_units, guaranteed),
                "borrowed_usage": max(0.0, usage.gpu_units - guaranteed),
                "unused_entitlement": max(0.0, guaranteed - usage.gpu_units),
                "fairshare_debt": self._debts.get(queue_id, 0.0),
                "historical_service": self.history.service.get(queue_id, 0.0),
                "normalized_entitlement": spec.weight / sibling_weight if sibling_weight else 1.0,
            }
        return result

    def metrics(self) -> dict[str, Any]:
        return {
            "allocation_policy": self.name,
            "historical_fairshare_enabled": self.historical,
        }
