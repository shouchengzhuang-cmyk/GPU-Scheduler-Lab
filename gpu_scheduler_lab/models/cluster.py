from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from gpu_scheduler_lab.models.job import Job


@dataclass(slots=True)
class GPU:
    id: str
    node_id: str
    memory_capacity_gb: float
    allocated_memory_gb: float = 0.0
    owner_job_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("gpu id must not be empty")
        if self.memory_capacity_gb <= 0:
            raise ValueError("GPU memory capacity must be positive")

    @property
    def occupied(self) -> bool:
        return self.owner_job_id is not None

    @property
    def free(self) -> bool:
        return not self.occupied

    def can_host(self, memory_gb: float) -> bool:
        return self.free and memory_gb <= self.memory_capacity_gb


@dataclass(slots=True)
class Node:
    id: str
    gpus: list[GPU]
    schedulable: bool = True
    topology: dict[str, str] = field(default_factory=dict)

    @property
    def occupied_gpu_count(self) -> int:
        return sum(gpu.occupied for gpu in self.gpus)

    @property
    def free_gpu_count(self) -> int:
        return sum(gpu.free for gpu in self.gpus)


@dataclass(slots=True)
class Cluster:
    nodes: list[Node]

    def __post_init__(self) -> None:
        node_ids = [node.id for node in self.nodes]
        gpu_ids = [gpu.id for node in self.nodes for gpu in node.gpus]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node ids must be unique")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("GPU ids must be unique")

    @property
    def gpus(self) -> list[GPU]:
        return [gpu for node in self.nodes for gpu in node.gpus]

    @property
    def total_gpu_count(self) -> int:
        return sum(len(node.gpus) for node in self.nodes)

    @property
    def total_memory_gb(self) -> float:
        return sum(gpu.memory_capacity_gb for gpu in self.gpus)

    def gpu_by_id(self, gpu_id: str) -> GPU:
        for gpu in self.gpus:
            if gpu.id == gpu_id:
                return gpu
        raise KeyError(gpu_id)

    def eligible_gpus(self, memory_gb: float) -> list[GPU]:
        return [
            gpu
            for node in self.nodes
            if node.schedulable
            for gpu in node.gpus
            if gpu.can_host(memory_gb)
        ]

    def allocate(self, job: Job, gpu_ids: Iterable[str]) -> None:
        selected_ids = list(gpu_ids)
        if len(selected_ids) != job.gpu_count or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("placement must contain exactly the requested number of unique GPUs")
        selected = [self.gpu_by_id(gpu_id) for gpu_id in selected_ids]
        if any(not gpu.can_host(job.gpu_memory_gb) for gpu in selected):
            raise ValueError("placement contains an unavailable or undersized GPU")
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

    def clone(self) -> Cluster:
        return Cluster(
            nodes=[
                Node(
                    id=node.id,
                    schedulable=node.schedulable,
                    topology=dict(node.topology),
                    gpus=[
                        GPU(
                            id=gpu.id,
                            node_id=gpu.node_id,
                            memory_capacity_gb=gpu.memory_capacity_gb,
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
                )
                for index, gpu_data in enumerate(node_data.get("gpus", []))
            ]
            nodes.append(
                Node(
                    id=node_id,
                    gpus=gpus,
                    schedulable=bool(node_data.get("schedulable", True)),
                    topology={str(k): str(v) for k, v in node_data.get("topology", {}).items()},
                )
            )
        return cls(nodes)
