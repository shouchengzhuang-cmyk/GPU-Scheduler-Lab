from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.models.cluster import GPU, Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.models.topology import TopologyMode, topology_domain
from gpu_scheduler_lab.queues.hierarchy import QueueHierarchy
from gpu_scheduler_lab.queues.model import ResourceVector


class AdmissionMode(StrEnum):
    PERMISSIVE = "permissive"
    QUOTA_AWARE = "quota-aware"


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    reason: str | None = None


class AdmissionController:
    def __init__(
        self,
        hierarchy: QueueHierarchy,
        cluster: Cluster,
        accounting: AccountingPolicy,
        mode: str = "permissive",
        potential_node_ids: set[str] | None = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.cluster = cluster
        self.accounting = accounting
        self.mode = AdmissionMode(mode)
        self.potential_node_ids = potential_node_ids if potential_node_ids is not None else set()

    def decide(self, job: Job) -> AdmissionDecision:
        if job.queue_id not in self.hierarchy.specs:
            return AdmissionDecision(False, "unknown_queue")
        minimum = job.minimum_gpu_count
        compatible = [
            gpu
            for node in self.cluster.nodes
            if (node.available and node.schedulable) or node.id in self.potential_node_ids
            for gpu in node.gpus
            if gpu.is_compatible(job)
        ]
        if len(compatible) < minimum:
            return AdmissionDecision(False, "impossible_gpu_request")
        feasible_sets = self._feasible_gpu_sets(job, compatible, minimum)
        if not feasible_sets:
            return AdmissionDecision(False, "impossible_gpu_request")
        if self.mode is AdmissionMode.QUOTA_AWARE:
            demand = min(
                (
                    self.accounting.minimum_demand(job, gpu_set, minimum)
                    for gpu_set in feasible_sets
                ),
                key=self._demand_key,
            )
            for ancestor_id in self.hierarchy.ancestors(job.queue_id):
                limit = self.hierarchy.specs[ancestor_id].limit
                if not demand.fits_within(limit):
                    return AdmissionDecision(False, "queue_hard_limit")
        return AdmissionDecision(True)

    def _feasible_gpu_sets(self, job: Job, compatible: list[GPU], minimum: int) -> list[list[GPU]]:
        groups: dict[str, list[GPU]]
        if job.topology_mode is TopologyMode.REQUIRE_SAME_NODE:
            groups = {}
            for gpu in compatible:
                groups.setdefault(gpu.node_id, []).append(gpu)
        elif job.topology_mode is TopologyMode.REQUIRE_SAME_RACK:
            nodes = {node.id: node for node in self.cluster.nodes}
            groups = {}
            for gpu in compatible:
                node = nodes[gpu.node_id]
                rack = topology_domain(node.id, node.topology, "rack")
                groups.setdefault(rack, []).append(gpu)
        else:
            groups = {"cluster": compatible}
        return [gpu_set for gpu_set in groups.values() if len(gpu_set) >= minimum]

    @staticmethod
    def _demand_key(demand: ResourceVector) -> tuple[float, float]:
        return demand.gpu_units, demand.gpu_memory_gb
