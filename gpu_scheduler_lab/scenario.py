from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job


@dataclass(slots=True)
class Scenario:
    cluster: Cluster
    jobs: list[Job]
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> Scenario:
        return Scenario(
            cluster=self.cluster.clone(),
            jobs=[job.clone() for job in self.jobs],
            metadata=dict(self.metadata),
        )


def load_scenario(path: Path) -> Scenario:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("scenario root must be a mapping")
    cluster = Cluster.from_dict(raw)
    jobs = [Job.from_dict(item) for item in raw.get("jobs", [])]
    return Scenario(cluster=cluster, jobs=jobs, metadata=dict(raw.get("metadata", {})))


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    return {
        "metadata": scenario.metadata,
        "nodes": [
            {
                "id": node.id,
                "schedulable": node.schedulable,
                "topology": node.topology,
                "gpus": [
                    {
                        "id": gpu.id,
                        "model": gpu.model,
                        "memory_gb": gpu.memory_capacity_gb,
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
                    {"topology_mode": job.topology_mode.value}
                    if job.topology_mode.value != "none"
                    else {}
                ),
                **({"checkpoint_cost": job.checkpoint_cost} if job.checkpoint_cost else {}),
                **({"restart_cost": job.restart_cost} if job.restart_cost else {}),
                **({"source_metadata": job.source_metadata} if job.source_metadata else {}),
            }
            for job in scenario.jobs
        ],
    }


def write_scenario(scenario: Scenario, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(scenario_to_dict(scenario), handle, sort_keys=False)
