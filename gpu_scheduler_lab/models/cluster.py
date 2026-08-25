from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from gpu_scheduler_lab.models.job import Job


@dataclass(slots=True)
class GPU:
    id: str
    node_id: str
    memory_capacity_gb: float
    model: str = "generic"
    allocated_memory_gb: float = 0.0
    owner_job_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("gpu id must not be empty")
        if not math.isfinite(self.memory_capacity_gb) or self.memory_capacity_gb <= 0:
            raise ValueError("GPU memory capacity must be positive")
        if not math.isfinite(self.allocated_memory_gb) or self.allocated_memory_gb < 0:
            raise ValueError("allocated GPU memory must be finite and non-negative")
        if not self.model:
            raise ValueError("GPU model must not be empty")

    @property
    def occupied(self) -> bool:
        return self.owner_job_id is not None

    @property
    def free(self) -> bool:
        return not self.occupied

    def is_compatible(self, request: Job | float) -> bool:
        if isinstance(request, Job):
            model_allowed = (request.gpu_model is None or self.model == request.gpu_model) and (
                not request.allowed_gpu_models or self.model in request.allowed_gpu_models
            )
            return model_allowed and request.gpu_memory_gb <= self.memory_capacity_gb
        return request <= self.memory_capacity_gb

    def can_host(self, request: Job | float) -> bool:
        return self.free and self.is_compatible(request)


@dataclass(slots=True)
class Node:
    id: str
    gpus: list[GPU]
    schedulable: bool = True
    topology: dict[str, str] = field(default_factory=dict)
    revocable: bool = False
    draining: bool = False
    available: bool = True

    @property
    def occupied_gpu_count(self) -> int:
        return sum(gpu.occupied for gpu in self.gpus)

    @property
    def free_gpu_count(self) -> int:
        return sum(gpu.free for gpu in self.gpus)


@dataclass(slots=True)
class Cluster:
    nodes: list[Node]
    _gpu_index: dict[str, GPU] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        node_ids = [node.id for node in self.nodes]
        gpu_ids = [gpu.id for node in self.nodes for gpu in node.gpus]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node ids must be unique")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("GPU ids must be unique")
        self._gpu_index = {gpu.id: gpu for node in self.nodes for gpu in node.gpus}
        for node in self.nodes:
            if any(gpu.node_id != node.id for gpu in node.gpus):
                raise ValueError(f"GPU node_id must match containing node {node.id}")

    @property
    def gpus(self) -> list[GPU]:
        return [gpu for node in self.nodes for gpu in node.gpus]

    @property
    def schedulable_nodes(self) -> list[Node]:
        return [node for node in self.nodes if node.schedulable and node.available]

    @property
    def active_nodes(self) -> list[Node]:
        """Capacity that exists now, including draining nodes with running work."""
        return [
            node for node in self.nodes if node.available and (node.schedulable or node.draining)
        ]

    @property
    def active_gpus(self) -> list[GPU]:
        return [gpu for node in self.active_nodes for gpu in node.gpus]

    @property
    def schedulable_gpus(self) -> list[GPU]:
        return [gpu for node in self.schedulable_nodes for gpu in node.gpus]

    @property
    def physical_gpu_count(self) -> int:
        return sum(len(node.gpus) for node in self.nodes)

    @property
    def physical_memory_gb(self) -> float:
        return sum(gpu.memory_capacity_gb for gpu in self.gpus)

    @property
    def total_gpu_count(self) -> int:
        """Usable GPU capacity; excludes GPUs on unschedulable nodes."""
        return len(self.schedulable_gpus)

    @property
    def total_memory_gb(self) -> float:
        """Usable memory capacity; excludes GPUs on unschedulable nodes."""
        return sum(gpu.memory_capacity_gb for gpu in self.schedulable_gpus)

    def gpu_by_id(self, gpu_id: str) -> GPU:
        try:
            return self._gpu_index[gpu_id]
        except KeyError as exc:
            raise KeyError(gpu_id) from exc

    def eligible_gpus(self, request: Job | float) -> list[GPU]:
        return [
            gpu for node in self.schedulable_nodes for gpu in node.gpus if gpu.can_host(request)
        ]

    def allocate(self, job: Job, gpu_ids: Iterable[str]) -> None:
        selected_ids = list(gpu_ids)
        if len(selected_ids) != job.requested_gpu_count or len(set(selected_ids)) != len(
            selected_ids
        ):
            raise ValueError("placement must contain exactly the requested number of unique GPUs")
        selected = [self.gpu_by_id(gpu_id) for gpu_id in selected_ids]
        schedulable_ids = {gpu.id for gpu in self.schedulable_gpus}
        if any(gpu.id not in schedulable_ids or not gpu.can_host(job) for gpu in selected):
            raise ValueError("placement contains an unavailable or undersized GPU")
        node_ids = [gpu.node_id for gpu in selected]
        topologies = {node.id: node.topology for node in self.nodes}
        from gpu_scheduler_lab.models.topology import topology_requirement_satisfied

        if not topology_requirement_satisfied(job.topology_mode, node_ids, topologies):
            raise ValueError("placement violates the job topology requirement")
        for gpu in selected:
            gpu.owner_job_id = job.id
            gpu.allocated_memory_gb = job.gpu_memory_gb
        job.allocated_gpu_ids = selected_ids

    def release(self, job: Job) -> None:
        for gpu_id in list(job.allocated_gpu_ids):
            gpu = self.gpu_by_id(gpu_id)
            if gpu.owner_job_id != job.id:
                raise RuntimeError(f"GPU ownership mismatch for {gpu.id}")
            gpu.owner_job_id = None
            gpu.allocated_memory_gb = 0.0
        job.allocated_gpu_ids.clear()

    def resize(self, job: Job, gpu_ids: Iterable[str]) -> None:
        target = list(gpu_ids)
        if len(set(target)) != len(target):
            raise ValueError("resize placement must contain unique GPUs")
        current = set(job.allocated_gpu_ids)
        target_set = set(target)
        selected = [self.gpu_by_id(gpu_id) for gpu_id in target]
        topologies = {node.id: node.topology for node in self.nodes}
        from gpu_scheduler_lab.models.topology import topology_requirement_satisfied

        if not topology_requirement_satisfied(
            job.topology_mode,
            (gpu.node_id for gpu in selected),
            topologies,
        ):
            raise ValueError("resize placement violates the job topology requirement")
        schedulable_ids = {gpu.id for gpu in self.schedulable_gpus}
        for gpu_id in sorted(target_set - current):
            gpu = self.gpu_by_id(gpu_id)
            if gpu_id not in schedulable_ids or not gpu.can_host(job):
                raise ValueError("resize contains an unavailable or undersized GPU")
        for gpu_id in sorted(current - target_set):
            gpu = self.gpu_by_id(gpu_id)
            if gpu.owner_job_id != job.id:
                raise RuntimeError(f"GPU ownership mismatch for {gpu.id}")
            gpu.owner_job_id = None
            gpu.allocated_memory_gb = 0.0
        for gpu_id in sorted(target_set - current):
            gpu = self.gpu_by_id(gpu_id)
            gpu.owner_job_id = job.id
            gpu.allocated_memory_gb = job.gpu_memory_gb
        job.allocated_gpu_ids = target

    def assert_invariants(self) -> None:
        seen: set[str] = set()
        for gpu in self.gpus:
            if not 0 <= gpu.allocated_memory_gb <= gpu.memory_capacity_gb:
                raise AssertionError(f"memory invariant violated on {gpu.id}")
            if gpu.owner_job_id is None and gpu.allocated_memory_gb != 0:
                raise AssertionError(f"unowned memory allocation on {gpu.id}")
            if gpu.id in seen:
                raise AssertionError(f"duplicate GPU id {gpu.id}")
            seen.add(gpu.id)

    def clone(self, *, preserve_allocations: bool = False) -> Cluster:
        return Cluster(
            nodes=[
                Node(
                    id=node.id,
                    schedulable=node.schedulable,
                    topology=dict(node.topology),
                    revocable=node.revocable,
                    draining=node.draining,
                    available=node.available,
                    gpus=[
                        GPU(
                            id=gpu.id,
                            node_id=gpu.node_id,
                            memory_capacity_gb=gpu.memory_capacity_gb,
                            model=gpu.model,
                            allocated_memory_gb=(
                                gpu.allocated_memory_gb if preserve_allocations else 0.0
                            ),
                            owner_job_id=gpu.owner_job_id if preserve_allocations else None,
                        )
                        for gpu in node.gpus
                    ],
                )
                for node in self.nodes
            ]
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cluster:
        nodes: list[Node] = []
        for node_data in data.get("nodes", []):
            node_id = str(node_data["id"])
            gpus = [
                GPU(
                    id=str(gpu_data.get("id", f"{node_id}-gpu-{index}")),
                    node_id=node_id,
                    memory_capacity_gb=float(gpu_data["memory_gb"]),
                    model=str(gpu_data.get("model", "generic")),
                )
                for index, gpu_data in enumerate(node_data.get("gpus", []))
            ]
            nodes.append(
                Node(
                    id=node_id,
                    gpus=gpus,
                    schedulable=bool(node_data.get("schedulable", True)),
                    topology={str(k): str(v) for k, v in node_data.get("topology", {}).items()},
                    revocable=bool(node_data.get("revocable", False)),
                    available=bool(node_data.get("available", True)),
                )
            )
        return cls(nodes)
