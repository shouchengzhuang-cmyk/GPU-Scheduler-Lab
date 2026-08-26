from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from gpu_scheduler_lab.models.cluster import GPU, Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.queues.model import ResourceVector


@dataclass(frozen=True, slots=True)
class AccountingPolicy:
    model_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not model or not math.isfinite(weight) or weight <= 0
            for model, weight in self.model_weights.items()
        ):
            raise ValueError("accounting model weights must be finite and positive")

    @classmethod
    def from_dict(cls, data: Any) -> AccountingPolicy:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("accounting must be a mapping")
        raw = data.get("model_weights", {})
        if not isinstance(raw, dict):
            raise ValueError("accounting.model_weights must be a mapping")
        return cls({str(model): float(weight) for model, weight in raw.items()})

    def allocation(
        self, job: Job, cluster: Cluster, gpu_ids: list[str] | None = None
    ) -> ResourceVector:
        selected = gpu_ids if gpu_ids is not None else job.allocated_gpu_ids
        units = sum(
            self.model_weights.get(cluster.gpu_by_id(gpu_id).model, 1.0) for gpu_id in selected
        )
        return ResourceVector(units, job.gpu_memory_gb * len(selected))

    def demand(self, job: Job, cluster: Cluster, replicas: int | None = None) -> ResourceVector:
        count = job.requested_gpu_count if replicas is None else replicas
        return self.minimum_demand(job, cluster.gpus, count)

    def minimum_demand(
        self,
        job: Job,
        gpus: Iterable[GPU],
        replicas: int,
    ) -> ResourceVector:
        weights = sorted(
            self.model_weights.get(gpu.model, 1.0) for gpu in gpus if gpu.is_compatible(job)
        )
        if len(weights) < replicas:
            raise ValueError("insufficient compatible GPUs for accounting demand")
        return ResourceVector(sum(weights[:replicas]), replicas * job.gpu_memory_gb)

    def capacity(self, cluster: Cluster) -> ResourceVector:
        return ResourceVector(
            sum(self.model_weights.get(gpu.model, 1.0) for gpu in cluster.schedulable_gpus),
            sum(gpu.memory_capacity_gb for gpu in cluster.schedulable_gpus),
        )
