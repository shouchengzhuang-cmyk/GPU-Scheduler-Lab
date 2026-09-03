from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.fleet.events import FleetEvent
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.queues.hierarchy import QueueHierarchy
from gpu_scheduler_lab.queues.model import QueueSpec


@dataclass(slots=True)
class Scenario:
    cluster: Cluster
    jobs: list[Job]
    metadata: dict[str, Any] = field(default_factory=dict)
    queues: tuple[QueueSpec, ...] = ()
    accounting: AccountingPolicy = field(default_factory=AccountingPolicy)
    admission_mode: str = "permissive"
    fairshare_half_life: float = 300.0
    starvation_threshold: float = 300.0
    fleet_events: tuple[FleetEvent, ...] = ()

    def clone(self) -> Scenario:
        return Scenario(
            cluster=self.cluster.clone(),
            jobs=[job.clone() for job in self.jobs],
            metadata=dict(self.metadata),
            queues=self.queues,
            accounting=self.accounting,
            admission_mode=self.admission_mode,
            fairshare_half_life=self.fairshare_half_life,
            starvation_threshold=self.starvation_threshold,
            fleet_events=self.fleet_events,
        )


def load_scenario(path: Path) -> Scenario:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("scenario root must be a mapping")
    cluster = Cluster.from_dict(raw)
    jobs = [Job.from_dict(item) for item in raw.get("jobs", [])]
    queues_raw = raw.get("queues", [])
    if not isinstance(queues_raw, list):
        raise ValueError("queues must be a list")
    admission = raw.get("admission", {})
    fairshare = raw.get("fairshare", {})
    if not isinstance(admission, dict) or not isinstance(fairshare, dict):
        raise ValueError("admission and fairshare must be mappings")
    events_raw = raw.get("fleet_events", [])
    if not isinstance(events_raw, list):
        raise ValueError("fleet_events must be a list")
    fleet_events = tuple(FleetEvent.from_dict(item) for item in events_raw)
    node_ids = {node.id for node in cluster.nodes}
    if any(event.node_id not in node_ids for event in fleet_events):
        raise ValueError("fleet event references an unknown node")
    queues = tuple(QueueSpec.from_dict(item) for item in queues_raw)
    QueueHierarchy(queues)
    half_life = float(fairshare.get("half_life", 300.0))
    starvation_threshold = float(fairshare.get("starvation_threshold", 300.0))
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError("fairshare.half_life must be finite and positive")
    if not math.isfinite(starvation_threshold) or starvation_threshold < 0:
        raise ValueError("fairshare.starvation_threshold must be finite and non-negative")
    return Scenario(
        cluster=cluster,
        jobs=jobs,
        metadata=dict(raw.get("metadata", {})),
        queues=queues,
        accounting=AccountingPolicy.from_dict(raw.get("accounting")),
        admission_mode=str(admission.get("mode", "permissive")),
        fairshare_half_life=half_life,
        starvation_threshold=starvation_threshold,
        fleet_events=fleet_events,
    )


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": scenario.metadata,
        "nodes": [
            {
                "id": node.id,
                "schedulable": node.schedulable,
                "topology": node.topology,
                **({"revocable": True} if node.revocable else {}),
                **({"available": False} if not node.available else {}),
                "gpus": [
                    {
                        "id": gpu.id,
                        "model": gpu.model,
                        "memory_gb": gpu.memory_capacity_gb,
                        **(
                            {
                                "vendor": gpu.vendor.value,
                                "kind": gpu.kind.value,
                                "runtime_profiles": list(gpu.runtime_profiles),
                                "capabilities": list(gpu.capabilities),
                            }
                            if not gpu.accelerator_metadata_inferred
                            else {}
                        ),
                    }
                    for gpu in node.gpus
                ],
            }
            for node in scenario.cluster.nodes
        ],
        "jobs": [
            {
                "id": job.id,
                "arrival_time": job.arrival_time,
                "duration": job.duration,
                "gpu_count": job.gpu_count,
                "gpu_memory_gb": job.gpu_memory_gb,
                "priority": job.priority.name.lower(),
                "type": job.job_type.value,
                "gang": job.gang,
                **({"sla_deadline": job.sla_deadline} if job.sla_deadline is not None else {}),
                **({"group": job.group} if job.group is not None else {}),
                **({"gpu_model": job.gpu_model} if job.gpu_model is not None else {}),
                **(
                    {"allowed_gpu_models": list(job.allowed_gpu_models)}
                    if job.allowed_gpu_models
                    else {}
                ),
                **(
                    {
                        "allowed_vendors": [value.value for value in job.allowed_vendors],
                        "allowed_kinds": [value.value for value in job.allowed_kinds],
                        "allowed_models": list(job.allowed_models),
                        "required_capabilities": list(job.required_capabilities),
                        "runtime_profile": job.runtime_profile,
                        "selection_policy": job.selection_policy.value,
                    }
                    if job.accelerator_request_explicit
                    else {}
                ),
                **(
                    {"topology_mode": job.topology_mode.value}
                    if job.topology_mode.value != "none"
                    else {}
                ),
                **({"checkpoint_cost": job.checkpoint_cost} if job.checkpoint_cost else {}),
                **({"restart_cost": job.restart_cost} if job.restart_cost else {}),
                **({"source_metadata": job.source_metadata} if job.source_metadata else {}),
                **({"queue": job.queue_id} if job.queue_id != "root/default" else {}),
                **({"elastic": job.elastic.to_dict()} if job.elastic is not None else {}),
            }
            for job in scenario.jobs
        ],
    }
    if scenario.queues:
        payload["queues"] = [queue.to_dict() for queue in scenario.queues]
    if scenario.accounting.model_weights:
        payload["accounting"] = {"model_weights": scenario.accounting.model_weights}
    if scenario.admission_mode != "permissive":
        payload["admission"] = {"mode": scenario.admission_mode}
    if scenario.fairshare_half_life != 300.0 or scenario.starvation_threshold != 300.0:
        payload["fairshare"] = {
            "half_life": scenario.fairshare_half_life,
            "starvation_threshold": scenario.starvation_threshold,
        }
    if scenario.fleet_events:
        payload["fleet_events"] = [event.to_dict() for event in scenario.fleet_events]
    return payload


def write_scenario(scenario: Scenario, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(scenario_to_dict(scenario), handle, sort_keys=False)
