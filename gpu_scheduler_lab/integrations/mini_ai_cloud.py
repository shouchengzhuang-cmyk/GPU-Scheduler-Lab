from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from gpu_scheduler_lab.models.accelerator import (
    AcceleratorKind,
    AcceleratorSelectionPolicy,
    AcceleratorVendor,
    vendor_supports_kind,
)
from gpu_scheduler_lab.models.cluster import GPU, Cluster, Node
from gpu_scheduler_lab.models.job import Job, JobType, Priority
from gpu_scheduler_lab.models.topology import TopologyMode
from gpu_scheduler_lab.scenario import Scenario

CONTRACT_V1_VERSION = "mini-ai-cloud.gpu-scheduler-lab/v1"
CONTRACT_V2_VERSION = "mini-ai-cloud.gpu-scheduler-lab/v2"
CONTRACT_VERSION = CONTRACT_V1_VERSION
RESULT_CONTRACT_VERSION = "gpu-scheduler-lab.result/v1"

_TOP_LEVEL_FIELDS = {"contract_version", "workers", "tasks", "exported_at", "producer"}
_WORKER_FIELDS = {"id", "schedulable", "labels", "gpu_devices"}
_DEVICE_V1_FIELDS = {"device_uuid", "memory_total_mb", "health", "model"}
_DEVICE_V2_FIELDS = _DEVICE_V1_FIELDS | {
    "vendor",
    "kind",
    "runtime_profiles",
    "capabilities",
}
_TASK_COMMON_FIELDS = {
    "id",
    "project_id",
    "arrival_time",
    "queued_at",
    "duration_seconds",
    "timeout_seconds",
    "gpu_count",
    "gpu_memory_mb",
    "priority",
    "workload_type",
    "sla_deadline",
    "labels",
}
_TASK_V1_FIELDS = _TASK_COMMON_FIELDS | {
    "gpu_model",
    "allowed_gpu_models",
}
_TASK_V2_FIELDS = _TASK_COMMON_FIELDS | {
    "allowed_vendors",
    "allowed_kinds",
    "allowed_models",
    "required_capabilities",
    "runtime_profile",
    "selection_policy",
}
_ISO_8601_TIMESTAMP = re.compile(
    r"^(?:(?:(?:(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26]))|"
    r"(?:(?:0[48]|[2468][048]|[13579][26])00))-02-29)|(?:(?!0000)[0-9]{4}-(?:(?:01|03|"
    r"05|07|08|10|12)-(?:0[1-9]|[12][0-9]|3[01])|(?:04|06|09|11)-(?:0[1-9]|"
    r"[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))))T(?:[01][0-9]|2[0-3]):[0-5][0-9]:"
    r"[0-5][0-9](?:\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?$"
)


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        limit = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{path} must be {limit}")
    return value


def _number(value: object, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{path} must be {qualifier}")
    return result


def _timestamp(value: object, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a number or ISO-8601 timestamp")
    if isinstance(value, int | float):
        return _number(value, path)
    if isinstance(value, str):
        if _ISO_8601_TIMESTAMP.fullmatch(value) is None:
            raise ValueError(f"{path} must be a valid ISO-8601 timestamp")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{path} must be a valid ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    raise ValueError(f"{path} must be a number or ISO-8601 timestamp")


def _priority(value: int) -> Priority:
    if value >= 90:
        return Priority.CRITICAL
    if value >= 75:
        return Priority.HIGH
    if value >= 25:
        return Priority.NORMAL
    return Priority.LOW


def _unknown_names(value: Mapping[str, Any], known: set[str]) -> list[str]:
    return sorted(set(value) - known)


def _unique_strings(value: object, path: str) -> list[str]:
    values = _list(value, path)
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(values)]
    if len(set(result)) != len(result):
        raise ValueError(f"{path} must not contain duplicates")
    return result


def validate_mini_ai_cloud_export(payload: object) -> dict[str, Any]:
    """Validate the v1/v2 compatibility surface without producer dependencies.

    Unknown fields are accepted for forward compatibility and audited during import.
    Every field consumed by the adapter is validated by this dependency-free runtime gate.
    """
    root = _mapping(payload, "export")
    contract_version = root.get("contract_version")
    if contract_version not in {CONTRACT_V1_VERSION, CONTRACT_V2_VERSION}:
        raise ValueError(
            f"contract_version must be one of {CONTRACT_V1_VERSION}, {CONTRACT_V2_VERSION}"
        )
    is_v2 = contract_version == CONTRACT_V2_VERSION
    workers = _list(root.get("workers"), "workers")
    tasks = _list(root.get("tasks"), "tasks")
    worker_ids: set[str] = set()
    gpu_ids: set[str] = set()
    task_ids: set[str] = set()

    for worker_index, worker_value in enumerate(workers):
        path = f"workers[{worker_index}]"
        worker = _mapping(worker_value, path)
        worker_id = _string(worker.get("id"), f"{path}.id")
        if worker_id in worker_ids:
            raise ValueError(f"duplicate worker id: {worker_id}")
        worker_ids.add(worker_id)
        if "schedulable" in worker and not isinstance(worker["schedulable"], bool):
            raise ValueError(f"{path}.schedulable must be a boolean")
        labels = _mapping(worker.get("labels", {}), f"{path}.labels")
        for key, value in labels.items():
            _string(key, f"{path}.labels key")
            _string(value, f"{path}.labels.{key}")
        devices = _list(worker.get("gpu_devices"), f"{path}.gpu_devices")
        for device_index, device_value in enumerate(devices):
            device_path = f"{path}.gpu_devices[{device_index}]"
            device = _mapping(device_value, device_path)
            device_id = str(device.get("device_uuid", f"{worker_id}-gpu-{device_index}"))
            _string(device_id, f"{device_path}.device_uuid")
            if device_id in gpu_ids:
                raise ValueError(f"duplicate GPU device_uuid: {device_id}")
            gpu_ids.add(device_id)
            _integer(device.get("memory_total_mb"), f"{device_path}.memory_total_mb", minimum=1)
            if "health" in device:
                _string(device["health"], f"{device_path}.health")
            if "model" in device:
                _string(device["model"], f"{device_path}.model")
            if is_v2:
                vendor = AcceleratorVendor(_string(device.get("vendor"), f"{device_path}.vendor"))
                kind = AcceleratorKind(_string(device.get("kind"), f"{device_path}.kind"))
                if not vendor_supports_kind(vendor, kind):
                    raise ValueError(f"{device_path}.vendor and kind must form a supported pair")
                _string(device.get("model"), f"{device_path}.model")
                _unique_strings(device.get("runtime_profiles"), f"{device_path}.runtime_profiles")
                _unique_strings(device.get("capabilities"), f"{device_path}.capabilities")

    for task_index, task_value in enumerate(tasks):
        path = f"tasks[{task_index}]"
        task = _mapping(task_value, path)
        task_id = _string(task.get("id"), f"{path}.id")
        if task_id in task_ids:
            raise ValueError(f"duplicate task id: {task_id}")
        task_ids.add(task_id)
        gpu_count = _integer(task.get("gpu_count"), f"{path}.gpu_count")
        if gpu_count > 0:
            _integer(task.get("gpu_memory_mb"), f"{path}.gpu_memory_mb", minimum=1)
        if "priority" in task:
            _integer(task["priority"], f"{path}.priority", maximum=100)
        for name in ("arrival_time", "queued_at", "sla_deadline"):
            if name in task:
                _timestamp(task[name], f"{path}.{name}")
        duration_value = task.get("duration_seconds", task.get("timeout_seconds", 60.0))
        _number(duration_value, f"{path}.duration_seconds", positive=True)
        if not is_v2:
            if "gpu_model" in task:
                _string(task["gpu_model"], f"{path}.gpu_model")
            legacy_allowed_models = _list(
                task.get("allowed_gpu_models", []), f"{path}.allowed_gpu_models"
            )
            for model_index, model in enumerate(legacy_allowed_models):
                _string(model, f"{path}.allowed_gpu_models[{model_index}]")
            if task.get("gpu_model") is not None and legacy_allowed_models:
                raise ValueError(f"{path}.gpu_model and allowed_gpu_models are mutually exclusive")
        if is_v2:
            allowed_vendors = _unique_strings(
                task.get("allowed_vendors"), f"{path}.allowed_vendors"
            )
            allowed_kinds = _unique_strings(task.get("allowed_kinds"), f"{path}.allowed_kinds")
            _unique_strings(task.get("allowed_models"), f"{path}.allowed_models")
            _unique_strings(task.get("required_capabilities"), f"{path}.required_capabilities")
            parsed_vendors: list[AcceleratorVendor] = []
            for index, vendor_value in enumerate(allowed_vendors):
                try:
                    parsed_vendors.append(AcceleratorVendor(vendor_value))
                except ValueError as exc:
                    raise ValueError(f"{path}.allowed_vendors[{index}] is unsupported") from exc
            parsed_kinds: list[AcceleratorKind] = []
            for index, kind_value in enumerate(allowed_kinds):
                try:
                    parsed_kinds.append(AcceleratorKind(kind_value))
                except ValueError as exc:
                    raise ValueError(f"{path}.allowed_kinds[{index}] is unsupported") from exc
            if (
                parsed_vendors
                and parsed_kinds
                and not any(
                    vendor_supports_kind(vendor, kind)
                    for vendor in parsed_vendors
                    for kind in parsed_kinds
                )
            ):
                raise ValueError(
                    f"{path}.allowed_vendors and allowed_kinds must include a supported pair"
                )
            runtime_profile = task.get("runtime_profile")
            if runtime_profile is not None:
                _string(runtime_profile, f"{path}.runtime_profile")
            try:
                AcceleratorSelectionPolicy(
                    _string(task.get("selection_policy"), f"{path}.selection_policy")
                )
            except ValueError as exc:
                raise ValueError(f"{path}.selection_policy is unsupported") from exc
        labels = _mapping(task.get("labels", {}), f"{path}.labels")
        for key, value in labels.items():
            _string(key, f"{path}.labels key")
            _string(value, f"{path}.labels.{key}")
        topology = labels.get("gpu_scheduler_lab/topology", "none")
        try:
            TopologyMode(str(topology))
        except ValueError as exc:
            raise ValueError(f"{path}.labels.gpu_scheduler_lab/topology is unsupported") from exc
    return dict(root)


def validate_result_handoff(payload: object) -> dict[str, Any]:
    root = _mapping(payload, "result")
    if root.get("contract_version") != RESULT_CONTRACT_VERSION:
        raise ValueError(f"result contract_version must be {RESULT_CONTRACT_VERSION}")
    if root.get("evidence_kind") != "SIMULATED":
        raise ValueError("result evidence_kind must be SIMULATED")
    limitations = _list(root.get("limitations"), "result.limitations")
    if not limitations:
        raise ValueError("result.limitations must contain at least one boundary")
    for index, limitation in enumerate(limitations):
        _string(limitation, f"result.limitations[{index}]")
    results = _list(root.get("results"), "result.results")
    for index, value in enumerate(results):
        item = _mapping(value, f"result.results[{index}]")
        _string(item.get("scheduler"), f"result.results[{index}].scheduler")
        elapsed = _number(item.get("elapsed_seconds"), f"result.results[{index}].elapsed_seconds")
        if elapsed < 0:
            raise ValueError(f"result.results[{index}].elapsed_seconds must be >= 0")
        _mapping(item.get("metrics"), f"result.results[{index}].metrics")
        _list(item.get("jobs"), f"result.results[{index}].jobs")
        if "trace" in item:
            _list(item["trace"], f"result.results[{index}].trace")
    return dict(root)


def import_mini_ai_cloud_export(payload: dict[str, Any]) -> Scenario:
    """Convert a stable file export without importing Mini-AI-Cloud internals."""
    validated = validate_mini_ai_cloud_export(payload)
    is_v2 = validated["contract_version"] == CONTRACT_V2_VERSION
    workers = validated["workers"]
    tasks = validated["tasks"]
    nodes: list[Node] = []
    ignored_worker_fields: set[str] = set()
    ignored_device_fields: set[str] = set()
    unhealthy_devices_filtered = 0
    for worker_value in workers:
        worker = _mapping(worker_value, "worker")
        worker_id = str(worker["id"])
        devices = []
        ignored_worker_fields.update(_unknown_names(worker, _WORKER_FIELDS))
        for index, device_value in enumerate(worker["gpu_devices"]):
            device = _mapping(device_value, "device")
            device_fields = _DEVICE_V2_FIELDS if is_v2 else _DEVICE_V1_FIELDS
            ignored_device_fields.update(_unknown_names(device, device_fields))
            if str(device.get("health", "healthy")) != "healthy":
                unhealthy_devices_filtered += 1
                continue
            devices.append(
                GPU(
                    id=str(device.get("device_uuid", f"{worker_id}-gpu-{index}")),
                    node_id=worker_id,
                    memory_capacity_gb=int(device["memory_total_mb"]) / 1024.0,
                    model=str(device.get("model", "generic")),
                    vendor=AcceleratorVendor(
                        str(device["vendor"]) if is_v2 else AcceleratorVendor.UNKNOWN.value
                    ),
                    kind=AcceleratorKind(
                        str(device["kind"]) if is_v2 else AcceleratorKind.GPU.value
                    ),
                    runtime_profiles=(
                        tuple(str(value) for value in device["runtime_profiles"]) if is_v2 else ()
                    ),
                    capabilities=(
                        tuple(str(value) for value in device["capabilities"]) if is_v2 else ()
                    ),
                    accelerator_metadata_inferred=not is_v2,
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

    raw_tasks = [task for task in tasks if int(_mapping(task, "task")["gpu_count"]) > 0]
    task_fields = _TASK_V2_FIELDS if is_v2 else _TASK_V1_FIELDS
    ignored_task_fields = {
        name
        for task_value in tasks
        for name in _unknown_names(_mapping(task_value, "task"), task_fields)
    }
    absolute_arrivals: list[float] = []
    for index, task_value in enumerate(raw_tasks):
        task = _mapping(task_value, "task")
        value = _timestamp(
            task.get("arrival_time", task.get("queued_at")), f"tasks[{index}].arrival_time"
        )
        if value is not None:
            absolute_arrivals.append(value)
    baseline = min(absolute_arrivals, default=0.0)
    jobs: list[Job] = []
    for index, task_value in enumerate(raw_tasks):
        task = _mapping(task_value, "task")
        ignored = _unknown_names(task, task_fields)
        arrival_value = _timestamp(
            task.get("arrival_time", task.get("queued_at")), f"tasks[{index}].arrival_time"
        )
        arrival = max(0.0, (baseline if arrival_value is None else arrival_value) - baseline)
        labels = _mapping(task.get("labels", {}), f"tasks[{index}].labels")
        deadline_value = _timestamp(task.get("sla_deadline"), f"tasks[{index}].sla_deadline")
        workload = str(task.get("workload_type", "batch_job"))
        gpu_count = int(task["gpu_count"])
        source_metadata = {"mini_ai_cloud_unknown_fields_ignored": ignored} if ignored else {}
        jobs.append(
            Job(
                id=str(task["id"]),
                arrival_time=arrival,
                duration=float(task.get("duration_seconds", task.get("timeout_seconds", 60.0))),
                gpu_count=gpu_count,
                gpu_memory_gb=int(task["gpu_memory_mb"]) / 1024.0,
                priority=_priority(int(task.get("priority", 50))),
                job_type=(
                    JobType.TRAINING if workload in {"training", "batch_job"} else JobType.INFERENCE
                ),
                gang=(
                    str(labels.get("gpu_scheduler_lab/gang", "")).lower() == "true" or gpu_count > 1
                ),
                sla_deadline=(
                    None if deadline_value is None else max(0.0, deadline_value - baseline)
                ),
                group=str(task.get("project_id", workload)),
                gpu_model=(
                    str(task["gpu_model"])
                    if not is_v2 and task.get("gpu_model") is not None
                    else None
                ),
                allowed_gpu_models=(
                    tuple(str(value) for value in task.get("allowed_gpu_models", []))
                    if not is_v2
                    else ()
                ),
                allowed_vendors=(
                    tuple(AcceleratorVendor(str(value)) for value in task["allowed_vendors"])
                    if is_v2
                    else ()
                ),
                allowed_kinds=(
                    tuple(AcceleratorKind(str(value)) for value in task["allowed_kinds"])
                    if is_v2
                    else ()
                ),
                allowed_models=(
                    tuple(str(value) for value in task["allowed_models"]) if is_v2 else ()
                ),
                required_capabilities=(
                    tuple(str(value) for value in task["required_capabilities"]) if is_v2 else ()
                ),
                runtime_profile=(
                    str(task["runtime_profile"])
                    if is_v2 and task["runtime_profile"] is not None
                    else None
                ),
                selection_policy=(
                    AcceleratorSelectionPolicy(str(task["selection_policy"]))
                    if is_v2
                    else AcceleratorSelectionPolicy.ANY
                ),
                accelerator_request_explicit=is_v2,
                topology_mode=TopologyMode(str(labels.get("gpu_scheduler_lab/topology", "none"))),
                source_metadata=source_metadata,
            )
        )
    return Scenario(
        cluster=Cluster(nodes),
        jobs=jobs,
        metadata={
            "source": "Mini-AI-Cloud",
            "contract_version": validated["contract_version"],
            "cpu_only_tasks_filtered": len(tasks) - len(raw_tasks),
            (
                "unhealthy_accelerator_devices_filtered"
                if is_v2
                else "unhealthy_gpu_devices_filtered"
            ): unhealthy_devices_filtered,
            "unknown_fields_ignored": {
                "export": _unknown_names(validated, _TOP_LEVEL_FIELDS),
                "worker": sorted(ignored_worker_fields),
                ("accelerator_device" if is_v2 else "gpu_device"): sorted(ignored_device_fields),
                "task": sorted(ignored_task_fields),
            },
        },
    )
