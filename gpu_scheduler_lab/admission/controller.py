from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
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
        if self.mode is AdmissionMode.QUOTA_AWARE:
            demand = self.accounting.minimum_demand(job, compatible, minimum)
            for ancestor_id in self.hierarchy.ancestors(job.queue_id):
                limit = self.hierarchy.specs[ancestor_id].limit
                if not demand.fits_within(limit):
                    return AdmissionDecision(False, "queue_hard_limit")
        return AdmissionDecision(True)
