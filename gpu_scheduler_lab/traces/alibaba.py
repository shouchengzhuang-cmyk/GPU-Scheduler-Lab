from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from gpu_scheduler_lab.models.cluster import GPU, Cluster, Node
from gpu_scheduler_lab.models.job import Job, JobType, Priority
from gpu_scheduler_lab.scenario import Scenario
from gpu_scheduler_lab.traces.base import TraceFilter
from gpu_scheduler_lab.traces.normalization import (
    NormalizationStats,
    deterministic_sample,
    normalize_timestamps,
)

ALIBABA_SPOT_GPU_SOURCE = (
    "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-spot-gpu"
)
ALIBABA_SPOT_GPU_VERSION = "cluster-trace-v2026-spot-gpu"

DEFAULT_GPU_MEMORY_GB = {
    "A10": 24.0,
    "A100-SXM4-40GB": 40.0,
    "A100-SXM4-80GB": 80.0,
    "H100-80GB": 80.0,
    "V100M16": 16.0,
    "V100M32": 32.0,
}


class AlibabaSpotGPUTraceAdapter:
    """Normalize Alibaba's 2026 spot-GPU CSV release into one Scenario."""

    def __init__(
        self,
        input_path: Path,
        *,
        gpu_memory_gb: dict[str, float] | None = None,
    ) -> None:
        self.input_path = input_path
        self.node_path, self.job_path = _resolve_paths(input_path)
        self.gpu_memory_gb = {**DEFAULT_GPU_MEMORY_GB, **(gpu_memory_gb or {})}
        self._stats = NormalizationStats()
        self._node_cpu_capacity: dict[str, float] = {}
        self._warnings: list[str] = []

    def to_scenario(self, trace_filter: TraceFilter) -> Scenario:
        cluster = self.load_cluster(trace_filter)
        jobs = self.load_jobs(trace_filter)
        return Scenario(cluster=cluster, jobs=jobs, metadata=self.metadata())

    def load_cluster(self, trace_filter: TraceFilter) -> Cluster:
        self._node_cpu_capacity = {}
        rows = _read_csv(self.node_path, {"node_name", "gpu_model", "gpu_capacity_num", "cpu_num"})
        rows.sort(key=lambda row: row["node_name"])
        if trace_filter.max_nodes is not None:
            rows = rows[: trace_filter.max_nodes]
        nodes: list[Node] = []
        for row_number, row in enumerate(rows, start=2):
            try:
                node_id = _required(row, "node_name")
                model = _required(row, "gpu_model")
                count = _positive_int(row, "gpu_capacity_num")
                memory = self._memory_for_model(model)
                cpu_capacity = _positive_float(row, "cpu_num")
            except ValueError as exc:
                raise ValueError(f"{self.node_path}:{row_number}: {exc}") from exc
            self._node_cpu_capacity[node_id] = cpu_capacity
            topology = {
                key: value for key in ("zone", "rack") if (value := row.get(key, "").strip())
            }
            nodes.append(
                Node(
                    id=node_id,
                    topology=topology,
                    gpus=[
                        GPU(
                            id=f"{node_id}-gpu-{index}",
                            node_id=node_id,
                            model=model,
                            memory_capacity_gb=memory,
                        )
                        for index in range(count)
                    ],
                )
            )
        return Cluster(nodes)

    def load_jobs(self, trace_filter: TraceFilter) -> list[Job]:
        required = {
            "job_name",
            "organization",
            "gpu_model",
            "cpu_request",
            "gpu_request",
            "worker_num",
            "submit_time",
            "duration",
            "job_type",
        }
        rows = _read_csv(self.job_path, required)
        self._stats = NormalizationStats(source_rows=len(rows))
        self._warnings = []
        parsed: list[tuple[float, str, dict[str, Any]]] = []
        window_end = (
            trace_filter.start + trace_filter.duration
            if trace_filter.duration is not None
            else math.inf
        )
        for row_number, row in enumerate(rows, start=2):
            try:
                item = self._parse_job_row(row)
            except ValueError as exc:
                if not trace_filter.skip_invalid:
                    raise ValueError(f"{self.job_path}:{row_number}: {exc}") from exc
                self._stats.invalid_rows += 1
                if len(self._warnings) < 20:
                    self._warnings.append(f"row {row_number}: {exc}")
                continue
            submit_time = float(item["submit_time"])
            if not trace_filter.start <= submit_time < window_end:
                self._stats.window_filtered_rows += 1
                continue
            identifier = str(item["id"])
            if not deterministic_sample(
                identifier, rate=trace_filter.sample_rate, seed=trace_filter.seed
            ):
                self._stats.sampled_rows += 1
                continue
            parsed.append((submit_time, identifier, item))
        parsed.sort(key=lambda item: (item[0], item[1]))
        if trace_filter.max_jobs is not None:
            parsed = parsed[: trace_filter.max_jobs]
        normalized, origin = normalize_timestamps(item[0] for item in parsed)
        self._stats.time_origin = origin
        jobs: list[Job] = []
        for arrival, (_, _, item) in zip(normalized, parsed, strict=True):
            jobs.append(
                Job(
                    id=str(item["id"]),
                    arrival_time=arrival,
                    duration=float(item["duration"]),
                    gpu_count=int(item["gpu_count"]),
                    gpu_memory_gb=float(item["gpu_memory_gb"]),
                    gpu_model=item["gpu_model"],
                    priority=item["priority"],
                    job_type=JobType.TRAINING,
                    gang=int(item["gpu_count"]) > 1,
                    group=str(item["organization"]),
                    source_metadata=dict(item["source_metadata"]),
                )
            )
        self._stats.selected_rows = len(jobs)
        return jobs

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "Alibaba Cluster Trace Program",
            "source_url": ALIBABA_SPOT_GPU_SOURCE,
            "trace_version": ALIBABA_SPOT_GPU_VERSION,
            "adapter": "alibaba-spot-gpu-v2026",
            "normalization": self._stats.to_dict(),
            "warnings": list(self._warnings),
            "node_cpu_capacity": dict(sorted(self._node_cpu_capacity.items())),
            "limitations": [
                "CPU requests are preserved as source metadata but are not placement constraints.",
                "GPU memory demand is inferred from the requested GPU model capacity.",
                "The public spot-GPU CSV does not expose rack or zone topology.",
            ],
        }

    def _parse_job_row(self, row: dict[str, str]) -> dict[str, Any]:
        identifier = _required(row, "job_name")
        model = row.get("gpu_model", "").strip() or None
        gpu_per_worker = _positive_int(row, "gpu_request")
        worker_count = _positive_int(row, "worker_num")
        gpu_count = gpu_per_worker * worker_count
        submit_time = _non_negative_float(row, "submit_time")
        duration = _positive_float(row, "duration")
        cpu_request = _non_negative_float(row, "cpu_request")
        if model is None:
            self._stats.jobs_without_gpu_model += 1
            gpu_memory = 1.0
        else:
            gpu_memory = self._memory_for_model(model)
        job_type = _required(row, "job_type").lower()
        if job_type in {"hp", "high-priority"}:
            priority = Priority.HIGH
        elif job_type in {"spot", "low-priority"}:
            priority = Priority.LOW
        else:
            raise ValueError(f"unsupported job_type {row['job_type']!r}")
        return {
            "id": identifier,
            "organization": _required(row, "organization"),
            "gpu_model": model,
            "gpu_count": gpu_count,
            "gpu_memory_gb": gpu_memory,
            "submit_time": submit_time,
            "duration": duration,
            "priority": priority,
            "source_metadata": {
                "cpu_request": cpu_request,
                "gpu_request_per_worker": gpu_per_worker,
                "worker_num": worker_count,
                "job_type": row["job_type"],
            },
        }

    def _memory_for_model(self, model: str) -> float:
        try:
            return self.gpu_memory_gb[model]
        except KeyError as exc:
            raise ValueError(
                f"GPU model {model!r} has no memory mapping; provide MODEL=GB explicitly"
            ) from exc


def _resolve_paths(input_path: Path) -> tuple[Path, Path]:
    if input_path.is_dir():
        node_path = input_path / "node_info_df.csv"
        job_path = input_path / "job_info_df.csv"
    else:
        raise ValueError("Alibaba trace input must be a directory containing both CSV files")
    missing = [str(path) for path in (node_path, job_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Alibaba trace files: {', '.join(missing)}")
    return node_path, job_path


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} must not be empty")
    return value


def _positive_int(row: dict[str, str], key: str) -> int:
    value = _positive_float(row, key)
    if not value.is_integer():
        raise ValueError(f"{key} must be an integer; fractional GPU is outside this simulator")
    return int(value)


def _positive_float(row: dict[str, str], key: str) -> float:
    value = _number(row, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _non_negative_float(row: dict[str, str], key: str) -> float:
    value = _number(row, key)
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _number(row: dict[str, str], key: str) -> float:
    raw = _required(row, key)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value
