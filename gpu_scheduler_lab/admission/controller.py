from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.models.cluster import GPU, Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.models.topology import TopologyMode, topology_domain
from gpu_scheduler_lab.queues.hierarchy import QueueHierarchy


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
        self.potential_node_ids = potential_node_ids or set()

    def decide(self, job: Job) -> AdmissionDecision:
        if job.queue_id not in self.hierarchy.specs:
            return AdmissionDecision(False, "unknown_queue")
        minimum = job.minimum_gpu_count
        compatible = [
            gpu
            for node in self.cluster.nodes
            if node.schedulable or node.id in self.potential_node_ids
            for gpu in node.gpus
            if gpu.is_compatible(job)
        ]
        if len(compatible) < minimum:
            return AdmissionDecision(False, "impossible_gpu_request")
        if not self._topology_feasible(job, compatible, minimum):
            return AdmissionDecision(False, "impossible_gpu_request")
        if self.mode is AdmissionMode.QUOTA_AWARE:
            demand = self.accounting.minimum_demand(job, compatible, minimum)
            for ancestor_id in self.hierarchy.ancestors(job.queue_id):
                limit = self.hierarchy.specs[ancestor_id].limit
                if not demand.fits_within(limit):
                    return AdmissionDecision(False, "queue_hard_limit")
        return AdmissionDecision(True)

    def _topology_feasible(self, job: Job, compatible: list[GPU], minimum: int) -> bool:
        if job.topology_mode is TopologyMode.REQUIRE_SAME_NODE:
            return any(
                sum(gpu.node_id == node.id for gpu in compatible) >= minimum
                for node in self.cluster.nodes
            )
        if job.topology_mode is TopologyMode.REQUIRE_SAME_RACK:
            nodes = {node.id: node for node in self.cluster.nodes}
            rack_counts: dict[str, int] = {}
            for gpu in compatible:
                node = nodes[gpu.node_id]
                rack = topology_domain(node.id, node.topology, "rack")
                rack_counts[rack] = rack_counts.get(rack, 0) + 1
            return any(count >= minimum for count in rack_counts.values())
        return True
