from __future__ import annotations

from typing import Any

from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.fairshare.drf import weighted_dominant_share
from gpu_scheduler_lab.fairshare.history import DecayedUsageHistory
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job, JobStatus
from gpu_scheduler_lab.queues.hierarchy import QueueHierarchy
from gpu_scheduler_lab.queues.model import ResourceVector
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.schedulers.topology import TopologyAwareScheduler


class FairShareScheduler(Scheduler):
    """Queue allocation policy composed with an independent placement scheduler."""

    dynamic_pending_order = True
    supports_guarantee_placement = True

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
        self._pending_branches: set[str] = set()

    def prepare(
        self,
        now: float,
        cluster: Cluster,
        pending: list[Job],
        running: list[Job],
    ) -> None:
        self.history.integrate(now, self._last_rates)
        self.refresh_usage(cluster, running)
        demanding_jobs = [
            *pending,
            *(
                job
                for job in running
                if job.status is JobStatus.RUNNING
                and job.elastic is not None
                and job.current_replicas < job.elastic.preferred_replicas
            ),
        ]
        self._pending_branches = {
            ancestor
            for job in demanding_jobs
            for ancestor in self.hierarchy.ancestors(job.queue_id)
            if ancestor != "root"
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
        self._sync_rates()

    def pending_key(self, job: Job, now: float) -> tuple[Any, ...]:
        path = tuple(
            self._queue_pending_key(queue_id)
            for queue_id in reversed(self.hierarchy.ancestors(job.queue_id))
            if queue_id != "root"
        )
        priority = int(job.priority) + sum(
            self.hierarchy.specs[queue_id].priority_offset
            for queue_id in self.hierarchy.ancestors(job.queue_id)
        )
        return (
            path,
            -priority,
            job.arrival_time,
            job.queue_id,
            job.id,
        )

    def _queue_pending_key(self, queue_id: str) -> tuple[float | int, ...]:
        spec = self.hierarchy.specs[queue_id]
        usage = self._aggregate_usage.get(queue_id, ResourceVector())
        dimensions = spec.guaranteed_dimensions or frozenset()
        guarantee_deficit = (
            "gpu_units" in dimensions and usage.gpu_units + 1e-9 < spec.guaranteed.gpu_units
        ) or (
            "gpu_memory_gb" in dimensions
            and usage.gpu_memory_gb + 1e-9 < spec.guaranteed.gpu_memory_gb
        )
        debt = self._debts.get(queue_id, 0.0) if self.historical else 0.0
        dominant = weighted_dominant_share(usage, self._capacity, spec.weight)
        return (
            0 if guarantee_deficit else 1,
            debt,
            dominant,
        )

    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        return self._place_with_borrowing(cluster, job, borrowing=self.borrowing)

    def place_guaranteed(self, cluster: Cluster, job: Job) -> list[str] | None:
        return self._place_with_borrowing(cluster, job, borrowing=False)

    def _place_with_borrowing(
        self,
        cluster: Cluster,
        job: Job,
        *,
        borrowing: bool,
    ) -> list[str] | None:
        placement = self.placement.place(cluster, job)
        if placement is None:
            return None
        actual_demand = self.accounting.allocation(job, cluster, placement)
        if not self.hierarchy.can_allocate(
            job.queue_id,
            actual_demand,
            self._direct_usage,
            borrowing=borrowing,
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
        self._sync_rates()
        self.placement.on_job_started(job, now)

    def _sync_rates(self) -> None:
        self._last_rates = {
            queue_id: usage.gpu_units for queue_id, usage in self._aggregate_usage.items()
        }

    def can_reclaim(self, victim: Job, incoming: Job) -> bool:
        boundary = self._reclaim_boundary(victim.queue_id, incoming.queue_id)
        if boundary is None or not self._boundary_has_reclaimable_surplus(*boundary):
            return False
        return all(
            self.hierarchy.specs[queue_id].reclaimable
            for queue_id in self.hierarchy.ancestors(victim.queue_id)
        )

    def can_reclaim_request(self, incoming: Job) -> bool:
        if self._cluster is None:
            return False
        try:
            demand = self.accounting.minimum_demand(
                incoming,
                self._cluster.schedulable_gpus,
                incoming.requested_gpu_count,
            )
        except ValueError:
            return False
        return self._demand_fits_entitlement(incoming.queue_id, demand)

    def can_reclaim_placement(
        self,
        incoming: Job,
        gpu_ids: list[str],
        entitlement_queue_ids: set[str] | None = None,
    ) -> bool:
        if self._cluster is None:
            return False
        demand = self.accounting.allocation(incoming, self._cluster, gpu_ids)
        return self._fits_remaining_entitlement(
            incoming.queue_id,
            demand,
            entitlement_queue_ids=entitlement_queue_ids,
        )

    def reclaim_entitlement_queue(self, victim: Job, incoming: Job) -> str | None:
        boundary = self._reclaim_boundary(victim.queue_id, incoming.queue_id)
        return boundary[1] if boundary is not None else None

    def can_reclaim_allocation(
        self,
        victim: Job,
        incoming: Job,
        released_gpu_ids: list[str],
        *,
        allow_indivisible_collateral: bool = False,
    ) -> bool:
        if self._cluster is None or not released_gpu_ids or not self.can_reclaim(victim, incoming):
            return False
        boundary = self._reclaim_boundary(victim.queue_id, incoming.queue_id)
        if boundary is None:
            return False
        victim_branch, _ = boundary
        spec = self.hierarchy.specs[victim_branch]
        usage = self._aggregate_usage.get(victim_branch, ResourceVector())
        released = self.accounting.allocation(victim, self._cluster, released_gpu_ids)
        dimensions = spec.guaranteed_dimensions or frozenset()
        if allow_indivisible_collateral:
            return True
        return all(
            getattr(usage, dimension) - getattr(released, dimension)
            >= getattr(spec.guaranteed, dimension) - 1e-9
            for dimension in dimensions
        )

    def can_scale_up(self, job: Job) -> bool:
        if not self.supports_elastic or job.elastic is None:
            return False
        return self._is_next_fairshare_branch(job.queue_id)

    def _is_next_fairshare_branch(self, queue_id: str) -> bool:
        path = [item for item in reversed(self.hierarchy.ancestors(queue_id)) if item != "root"]
        for branch in path:
            parent = self.hierarchy.specs[branch].parent or "root"
            contenders = [
                sibling
                for sibling in self.hierarchy.children[parent]
                if sibling == branch or sibling in self._pending_branches
            ]
            if len(contenders) <= 1:
                continue
            branch_key = self._queue_pending_key(branch)
            if branch_key > min(self._queue_pending_key(sibling) for sibling in contenders):
                return False
        return True

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

    def _fits_remaining_entitlement(
        self,
        queue_id: str,
        demand: ResourceVector,
        *,
        entitlement_queue_ids: set[str] | None = None,
    ) -> bool:
        found = False
        ancestors = self.hierarchy.ancestors(queue_id)
        candidates = (
            ancestors
            if entitlement_queue_ids is None
            else tuple(
                ancestor_id for ancestor_id in ancestors if ancestor_id in entitlement_queue_ids
            )
        )
        for ancestor_id in candidates:
            spec = self.hierarchy.specs[ancestor_id]
            dimensions = spec.guaranteed_dimensions or frozenset()
            if not dimensions:
                continue
            found = True
            projected = self._aggregate_usage.get(ancestor_id, ResourceVector()) + demand
            if (
                "gpu_units" in dimensions and projected.gpu_units > spec.guaranteed.gpu_units + 1e-9
            ) or (
                "gpu_memory_gb" in dimensions
                and projected.gpu_memory_gb > spec.guaranteed.gpu_memory_gb + 1e-9
            ):
                return False
        return found

    def _demand_fits_entitlement(self, queue_id: str, demand: ResourceVector) -> bool:
        found = False
        for ancestor_id in self.hierarchy.ancestors(queue_id):
            spec = self.hierarchy.specs[ancestor_id]
            dimensions = spec.guaranteed_dimensions or frozenset()
            if not dimensions:
                continue
            found = True
            if (
                "gpu_units" in dimensions and demand.gpu_units > spec.guaranteed.gpu_units + 1e-9
            ) or (
                "gpu_memory_gb" in dimensions
                and demand.gpu_memory_gb > spec.guaranteed.gpu_memory_gb + 1e-9
            ):
                return False
        return found

    def _reclaim_boundary(
        self,
        victim_queue_id: str,
        incoming_queue_id: str,
    ) -> tuple[str, str] | None:
        victim_path = self.hierarchy.ancestors(victim_queue_id)
        incoming_path = self.hierarchy.ancestors(incoming_queue_id)
        incoming_ancestors = set(incoming_path)
        common_ancestor = next(
            queue_id for queue_id in victim_path if queue_id in incoming_ancestors
        )
        victim_index = victim_path.index(common_ancestor)
        incoming_index = incoming_path.index(common_ancestor)
        if victim_index == 0 or incoming_index == 0:
            return None
        return victim_path[victim_index - 1], incoming_path[incoming_index - 1]

    def _boundary_has_reclaimable_surplus(
        self,
        victim_branch: str,
        incoming_branch: str,
    ) -> bool:
        incoming_spec = self.hierarchy.specs[incoming_branch]
        incoming_usage = self._aggregate_usage.get(incoming_branch, ResourceVector())
        victim_spec = self.hierarchy.specs[victim_branch]
        victim_usage = self._aggregate_usage.get(victim_branch, ResourceVector())
        incoming_dimensions = incoming_spec.guaranteed_dimensions or frozenset()
        victim_dimensions = victim_spec.guaranteed_dimensions or frozenset()
        for dimension in incoming_dimensions:
            incoming_value = getattr(incoming_usage, dimension)
            incoming_guarantee = getattr(incoming_spec.guaranteed, dimension)
            victim_value = getattr(victim_usage, dimension)
            victim_floor = (
                getattr(victim_spec.guaranteed, dimension)
                if dimension in victim_dimensions
                else 0.0
            )
            if incoming_value + 1e-9 < incoming_guarantee and victim_value > victim_floor + 1e-9:
                return True
        return False

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
