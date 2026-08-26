from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.scenario import Scenario


@dataclass(frozen=True, slots=True)
class TraceFilter:
    start: float = 0.0
    duration: float | None = None
    max_jobs: int | None = None
    max_nodes: int | None = None
    sample_rate: float = 1.0
    seed: int = 0
    skip_invalid: bool = False

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("trace start must be non-negative")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("trace duration must be positive")
        if self.max_jobs is not None and self.max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        if self.max_nodes is not None and self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if not 0 < self.sample_rate <= 1:
            raise ValueError("sample_rate must be in (0, 1]")


class TraceAdapter(Protocol):
    def load_cluster(self, trace_filter: TraceFilter) -> Cluster: ...

    def load_jobs(self, trace_filter: TraceFilter) -> list[Job]: ...

    def metadata(self) -> dict[str, Any]: ...


def scenario_from_adapter(adapter: TraceAdapter, trace_filter: TraceFilter) -> Scenario:
    cluster = adapter.load_cluster(trace_filter)
    jobs = adapter.load_jobs(trace_filter)
    return Scenario(cluster=cluster, jobs=jobs, metadata=adapter.metadata())
