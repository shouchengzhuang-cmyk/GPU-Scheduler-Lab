from __future__ import annotations

from datetime import datetime
from typing import Any

from gpu_scheduler_lab.models.cluster import GPU, Cluster, Node
from gpu_scheduler_lab.models.job import Job, JobType, Priority
from gpu_scheduler_lab.scenario import Scenario


def _priority(value: int) -> Priority:
    if value >= 90:
        return Priority.CRITICAL
    if value >= 75:
        return Priority.HIGH
    if value >= 25:
        return Priority.NORMAL
    return Priority.LOW


def _timestamp(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    raise ValueError(f"unsupported timestamp value: {value!r}")


def import_mini_ai_cloud_export(payload: dict[str, Any]) -> Scenario:
    """Convert a stable file export without importing Mini-AI-Cloud internals."""
    if payload.get("contract_version") != "mini-ai-cloud.gpu-scheduler-lab/v1":
        raise ValueError("contract_version must be mini-ai-cloud.gpu-scheduler-lab/v1")
    nodes: list[Node] = []
    for worker in payload.get("workers", []):
        worker_id = str(worker["id"])
        devices = []
        for index, device in enumerate(worker.get("gpu_devices", [])):
            if str(device.get("health", "healthy")) != "healthy":
                continue
            total_mb = int(device["memory_total_mb"])
            devices.append(
                GPU(
                    id=str(device.get("device_uuid", f"{worker_id}-gpu-{index}")),
                    node_id=worker_id,
                    memory_capacity_gb=total_mb / 1024.0,
                )
            )
        nodes.append(
            Node(
                id=worker_id,
                gpus=devices,
                schedulable=bool(worker.get("schedulable", True)),
                topology={str(k): str(v) for k, v in worker.get("labels", {}).items()},
            )
        )

    raw_tasks = [task for task in payload.get("tasks", []) if int(task.get("gpu_count", 0)) > 0]
    absolute_arrivals = [
        value
        for task in raw_tasks
        if (value := _timestamp(task.get("arrival_time", task.get("queued_at")))) is not None
    ]
    baseline = min(absolute_arrivals, default=0.0)
    jobs: list[Job] = []
    for task in raw_tasks:
        arrival_value = _timestamp(task.get("arrival_time", task.get("queued_at")))
        arrival = max(0.0, (arrival_value or baseline) - baseline)
        duration = float(task.get("duration_seconds", task.get("timeout_seconds", 60.0)))
        memory_mb = int(task.get("gpu_memory_mb", 0))
        if memory_mb <= 0:
            raise ValueError(f"GPU task {task.get('id')} must provide positive gpu_memory_mb")
        labels = task.get("labels", {})
        workload = str(task.get("workload_type", "batch_job"))
        training = workload in {"training", "batch_job"}
        gang_label = str(labels.get("gpu_scheduler_lab/gang", "")).lower() == "true"
        deadline_value = _timestamp(task.get("sla_deadline"))
        deadline = None if deadline_value is None else max(0.0, deadline_value - baseline)
        jobs.append(
            Job(
                id=str(task["id"]),
                arrival_time=arrival,
                duration=duration,
                gpu_count=int(task["gpu_count"]),
                gpu_memory_gb=memory_mb / 1024.0,
                priority=_priority(int(task.get("priority", 50))),
                job_type=JobType.TRAINING if training else JobType.INFERENCE,
                gang=gang_label or int(task["gpu_count"]) > 1,
                sla_deadline=deadline,
                group=str(task.get("project_id", workload)),
            )
        )
    return Scenario(
        cluster=Cluster(nodes),
        jobs=jobs,
        metadata={
            "source": "Mini-AI-Cloud",
            "contract_version": payload["contract_version"],
            "cpu_only_tasks_filtered": len(payload.get("tasks", [])) - len(raw_tasks),
        },
    )
