from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from gpu_scheduler_lab.elastic.work import ElasticSpec
from gpu_scheduler_lab.models.topology import TopologyMode


class Priority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

    @classmethod
    def parse(cls, value: str | Priority) -> Priority:
        if isinstance(value, Priority):
            return value
        try:
            return cls[value.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown priority: {value}") from exc


class JobType(StrEnum):
    INFERENCE = "inference"
    TRAINING = "training"


class JobStatus(StrEnum):
    REJECTED = "rejected"
    PENDING = "pending"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    RESTARTING = "restarting"
    COMPLETED = "completed"


@dataclass(slots=True)
class Job:
    id: str
    arrival_time: float
    duration: float
    gpu_count: int
    gpu_memory_gb: float
    priority: Priority = Priority.NORMAL
    job_type: JobType = JobType.INFERENCE
    gang: bool = False
    sla_deadline: float | None = None
    group: str | None = None
    gpu_model: str | None = None
    allowed_gpu_models: tuple[str, ...] = ()
    topology_mode: TopologyMode = TopologyMode.NONE
    checkpoint_cost: float = 0.0
    restart_cost: float = 0.0
    source_metadata: dict[str, Any] = field(default_factory=dict)
    queue_id: str = "root/default"
    elastic: ElasticSpec | None = None
    status: JobStatus = field(default=JobStatus.PENDING, init=False)
    allocated_gpu_ids: list[str] = field(default_factory=list, init=False)
    accumulated_runtime: float = field(default=0.0, init=False)
    last_start_time: float | None = field(default=None, init=False)
    first_start_time: float | None = field(default=None, init=False)
    completion_time: float | None = field(default=None, init=False)
    preemption_count: int = field(default=0, init=False)
    run_generation: int = field(default=0, init=False)
    running_priority: int | None = field(default=None, init=False)
    checkpoint_overhead: float = field(default=0.0, init=False)
    restart_overhead: float = field(default=0.0, init=False)
    admission_time: float | None = field(default=None, init=False)
    rejection_reason: str | None = field(default=None, init=False)
    productive_work_completed: float = field(default=0.0, init=False)
    current_replicas: int = field(default=0, init=False)
    requested_replicas: int = field(default=0, init=False)
    borrowed_gpu_units: float = field(default=0.0, init=False)
    reclaim_victim_count: int = field(default=0, init=False)
    recovery_count: int = field(default=0, init=False)
    recovery_overhead: float = field(default=0.0, init=False)
    elastic_scale_up_count: int = field(default=0, init=False)
    elastic_scale_down_count: int = field(default=0, init=False)
    resize_churn_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.priority = Priority.parse(self.priority)
        if isinstance(self.job_type, str):
            self.job_type = JobType(self.job_type)
        if isinstance(self.topology_mode, str):
            self.topology_mode = TopologyMode(self.topology_mode)
        self.allowed_gpu_models = tuple(self.allowed_gpu_models)
        if not self.id:
            raise ValueError("job id must not be empty")
        if not self.queue_id:
            raise ValueError("queue_id must not be empty")
        finite_values = {
            "arrival_time": self.arrival_time,
            "duration": self.duration,
            "gpu_memory_gb": self.gpu_memory_gb,
            "checkpoint_cost": self.checkpoint_cost,
            "restart_cost": self.restart_cost,
        }
        if self.sla_deadline is not None:
            finite_values["sla_deadline"] = self.sla_deadline
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        if self.gpu_memory_gb <= 0:
            raise ValueError("gpu_memory_gb must be positive")
        if self.sla_deadline is not None and self.sla_deadline < self.arrival_time:
            raise ValueError("sla_deadline must not precede arrival_time")
        if self.gpu_model is not None and self.allowed_gpu_models:
            raise ValueError("gpu_model and allowed_gpu_models are mutually exclusive")
        if self.gpu_model == "":
            raise ValueError("gpu_model must not be empty")
        if any(not model for model in self.allowed_gpu_models):
            raise ValueError("allowed_gpu_models must not contain empty values")
        if self.checkpoint_cost < 0 or self.restart_cost < 0:
            raise ValueError("preemption costs must be non-negative")
        self.requested_replicas = (
            self.elastic.preferred_replicas if self.elastic is not None else self.gpu_count
        )

    @property
    def minimum_gpu_count(self) -> int:
        return self.elastic.min_replicas if self.elastic is not None else self.gpu_count

    @property
    def preferred_gpu_count(self) -> int:
        return self.elastic.preferred_replicas if self.elastic is not None else self.gpu_count

    @property
    def maximum_gpu_count(self) -> int:
        return self.elastic.max_replicas if self.elastic is not None else self.gpu_count

    @property
    def requested_gpu_count(self) -> int:
        return self.requested_replicas or self.gpu_count

    @property
    def total_productive_work(self) -> float:
        replicas = self.preferred_gpu_count if self.elastic is not None else self.gpu_count
        return self.duration * replicas

    def productive_rate(self, replicas: int | None = None) -> float:
        count = self.current_replicas if replicas is None else replicas
        if self.elastic is None:
            return float(self.gpu_count)
        return self.elastic.work_rate(count)

    @property
    def remaining_duration(self) -> float:
        return max(0.0, self.duration - self.accumulated_runtime)

    @property
    def waiting_time(self) -> float | None:
        if self.completion_time is None:
            return None
        if self.elastic is not None:
            return (
                max(0.0, self.first_start_time - self.arrival_time)
                if self.first_start_time is not None
                else None
            )
        return max(0.0, self.completion_time - self.arrival_time - self.duration)

    @property
    def turnaround_time(self) -> float | None:
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time

    def effective_priority(self, now: float, aging_interval: float) -> int:
        if self.status is JobStatus.RUNNING:
            return (
                self.running_priority if self.running_priority is not None else int(self.priority)
            )
        if aging_interval <= 0:
            return int(self.priority)
        waited = max(0.0, now - self.arrival_time - self.accumulated_runtime)
        return min(int(Priority.CRITICAL), int(self.priority) + int(waited // aging_interval))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        return cls(
            id=str(data["id"]),
            arrival_time=float(data["arrival_time"]),
            duration=float(data["duration"]),
            gpu_count=int(data["gpu_count"]),
            gpu_memory_gb=float(data["gpu_memory_gb"]),
            priority=Priority.parse(str(data.get("priority", "normal"))),
            job_type=JobType(str(data.get("type", "inference"))),
            gang=bool(data.get("gang", False)),
            sla_deadline=(
                float(data["sla_deadline"]) if data.get("sla_deadline") is not None else None
            ),
            group=str(data["group"]) if data.get("group") is not None else None,
            gpu_model=str(data["gpu_model"]) if data.get("gpu_model") is not None else None,
            allowed_gpu_models=tuple(str(value) for value in data.get("allowed_gpu_models", [])),
            topology_mode=TopologyMode(str(data.get("topology_mode", "none"))),
            checkpoint_cost=float(data.get("checkpoint_cost", 0.0)),
            restart_cost=float(data.get("restart_cost", 0.0)),
            source_metadata=dict(data.get("source_metadata", {})),
            queue_id=str(data.get("queue", "root/default")),
            elastic=ElasticSpec.from_dict(data.get("elastic")),
        )

    def clone(self) -> Job:
        return Job(
            id=self.id,
            arrival_time=self.arrival_time,
            duration=self.duration,
            gpu_count=self.gpu_count,
            gpu_memory_gb=self.gpu_memory_gb,
            priority=self.priority,
            job_type=self.job_type,
            gang=self.gang,
            sla_deadline=self.sla_deadline,
            group=self.group,
            gpu_model=self.gpu_model,
            allowed_gpu_models=self.allowed_gpu_models,
            topology_mode=self.topology_mode,
            checkpoint_cost=self.checkpoint_cost,
            restart_cost=self.restart_cost,
            source_metadata=dict(self.source_metadata),
            queue_id=self.queue_id,
            elastic=self.elastic,
        )
