from __future__ import annotations

from gpu_scheduler_lab.models.accelerator import AcceleratorVendor
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.schedulers.topology import TopologyAwareScheduler


class VendorPreferenceScheduler(Scheduler):
    """Prefer one vendor while preserving typed feasibility and single-vendor gangs."""

    def __init__(self, preferred_vendor: AcceleratorVendor) -> None:
        if preferred_vendor is AcceleratorVendor.UNKNOWN:
            raise ValueError("preferred vendor must be explicit")
        self.preferred_vendor = preferred_vendor
        self.name = f"prefer-{preferred_vendor.value.replace('huawei-', '')}"
        self.placement = TopologyAwareScheduler()

    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        available_vendors = {
            gpu.vendor
            for gpu in cluster.schedulable_gpus
            if gpu.vendor is not AcceleratorVendor.UNKNOWN and gpu.is_compatible(job)
        }
        ordered = [
            self.preferred_vendor,
            *sorted(
                available_vendors - {self.preferred_vendor},
                key=lambda vendor: vendor.value,
            ),
        ]
        for vendor in ordered:
            if vendor not in available_vendors:
                continue
            placement = self.placement.place(
                cluster.clone(preserve_allocations=True, vendor=vendor),
                job,
            )
            if placement is not None:
                return placement
        return None
