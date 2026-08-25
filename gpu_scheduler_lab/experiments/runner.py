from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_scheduler_lab.allocation.allocator import FairShareScheduler
from gpu_scheduler_lab.experiments.aggregation import aggregate_runs
from gpu_scheduler_lab.experiments.config import ExperimentConfig
from gpu_scheduler_lab.experiments.manifest import git_sha, python_version, scenario_hash
from gpu_scheduler_lab.fleet.events import FleetEventType
from gpu_scheduler_lab.queues.hierarchy import QueueHierarchy
from gpu_scheduler_lab.queues.model import QueueSpec, ResourceVector
from gpu_scheduler_lab.scenario import Scenario, load_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.simulator.engine import Simulator
from gpu_scheduler_lab.visualization.experiment import plot_experiment_summary
from gpu_scheduler_lab.visualization.phase3 import plot_phase3_timelines
from gpu_scheduler_lab.workload import GeneratorConfig, generate_scenario


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    manifest: Path
    runs: Path
    summary_csv: Path
    summary_json: Path
    comparison: Path


def run_experiment(config_path: Path) -> ExperimentArtifacts:
    config = ExperimentConfig.load(config_path)
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    revision = git_sha(config_path.parent)
    runs: list[dict[str, Any]] = []
    for seed in config.seeds:
        scenario = _scenario_for_seed(config, seed)
        fingerprint = scenario_hash(scenario)
        for scheduler_name in config.schedulers:
            scheduler = _scheduler_for_run(config, scheduler_name, scenario)
            result = Simulator(
                scenario.cluster,
                scenario.jobs,
                scheduler,
                scenario=scenario,
            ).run()
            result_payload = result.to_dict(include_trace=True)
            result_payload.pop("metrics", None)
            runs.append(
                {
                    "scheduler": scheduler_name,
                    "seed": seed,
                    "scenario_hash": fingerprint,
                    "metrics": result.metrics,
                    "result": result_payload,
                }
            )
    summary = aggregate_runs(runs)
    manifest_runs = [
        {
            "scheduler": run["scheduler"],
            "seed": run["seed"],
            "scenario_hash": run["scenario_hash"],
            "metrics": _manifest_metrics(run["metrics"]),
        }
        for run in runs
    ]
    identity = hashlib.sha256(
        json.dumps(
            {
                "name": config.name,
                "schedulers": config.schedulers,
                "seeds": config.seeds,
                "workload": config.workload,
                "allocation_policy": config.allocation_policy,
                "placement_scheduler": config.placement_scheduler,
                "queue_policy": config.queue_policy,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    trace_metadata = _trace_metadata(config, runs)
    scenario_hashes = sorted({str(run["scenario_hash"]) for run in runs})
    representative = _scenario_for_seed(config, config.seeds[0])
    manifest = {
        "experiment_id": identity,
        "name": config.name,
        "timestamp": timestamp,
        "git_sha": revision,
        "python_version": python_version(),
        "scenario_hash": scenario_hashes[0] if len(scenario_hashes) == 1 else None,
        "scenario_hashes": scenario_hashes,
        "config": {
            "workload": config.workload,
            "schedulers": list(config.schedulers),
            "seeds": list(config.seeds),
            "allocation_policy": config.allocation_policy,
            "placement_scheduler": config.placement_scheduler,
            "queue_policy": config.queue_policy,
        },
        "queue_config_hash": _stable_hash([queue.to_dict() for queue in representative.queues]),
        "allocation_policy": config.allocation_policy or {"types": list(config.schedulers)},
        "fairshare_config": {
            "half_life": representative.fairshare_half_life,
            "starvation_threshold": representative.starvation_threshold,
        },
        "fleet_event_hash": _stable_hash(
            [event.to_dict() for event in representative.fleet_events]
        ),
        "elastic_model_version": "ideal-linear-v1",
        **trace_metadata,
        "runs": manifest_runs,
    }
    artifacts = ExperimentArtifacts(
        manifest=output / "manifest.json",
        runs=output / "runs.json",
        summary_csv=output / "summary.csv",
        summary_json=output / "summary.json",
        comparison=output / "comparison.png",
    )
    _write_json(artifacts.manifest, manifest)
    _write_json(artifacts.runs, {"runs": runs})
    _write_json(artifacts.summary_json, {"summary": summary})
    _write_summary_csv(artifacts.summary_csv, summary)
    plot_experiment_summary(summary, artifacts.comparison)
    plot_phase3_timelines(runs, output)
    return artifacts


def _scheduler_for_run(
    config: ExperimentConfig, scheduler_name: str, scenario: Scenario
) -> Scheduler:
    if not config.allocation_policy:
        return create_scheduler(scheduler_name, scenario)
    placement_name = config.placement_scheduler or "topology"
    placement = create_scheduler(placement_name, scenario)
    queue_policy = config.queue_policy
    historical = scheduler_name == "historical-drf"
    return FairShareScheduler(
        QueueHierarchy(scenario.queues),
        scenario.accounting,
        placement=placement,
        historical=historical,
        half_life=float(config.allocation_policy.get("half_life", scenario.fairshare_half_life)),
        borrowing=bool(queue_policy.get("borrowing", True)),
        reclaim=bool(queue_policy.get("reclaim", False)),
        elastic=bool(queue_policy.get("elastic", True)),
        name=scheduler_name,
    )


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _manifest_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"queue_timeline", "elastic_replica_timeline", "fleet_capacity_timeline"}
    }


def _scenario_for_seed(config: ExperimentConfig, seed: int) -> Scenario:
    workload_type = str(config.workload.get("type", "scenario"))
    if workload_type in {"scenario", "trace"}:
        raw_path = config.workload.get("scenario")
        if raw_path is None:
            raise ValueError("trace/scenario workload requires workload.scenario")
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        scenario = load_scenario(path)
        return _apply_tenant_overlay(scenario, config.workload)
    raw_generator = config.workload.get("generator", {})
    if not isinstance(raw_generator, dict):
        raise ValueError("workload.generator must be a mapping")
    values = dict(raw_generator)
    values["seed"] = seed
    scenario = generate_scenario(GeneratorConfig(**values))
    return _apply_tenant_overlay(scenario, config.workload)


def _apply_tenant_overlay(scenario: Scenario, workload: dict[str, Any]) -> Scenario:
    tenant_count = int(workload.get("tenant_count", 0))
    if tenant_count <= 0:
        return scenario
    potential_node_ids = {node.id for node in scenario.cluster.schedulable_nodes}
    potential_node_ids.update(
        event.node_id
        for event in scenario.fleet_events
        if event.event_type
        in {
            FleetEventType.NODE_JOIN,
            FleetEventType.NODE_RECOVER,
            FleetEventType.CAPACITY_RETURN,
        }
    )
    potential_gpus = [
        gpu for node in scenario.cluster.nodes if node.id in potential_node_ids for gpu in node.gpus
    ]
    total_gpu_units = sum(
        scenario.accounting.model_weights.get(gpu.model, 1.0) for gpu in potential_gpus
    )
    total_memory_gb = sum(gpu.memory_capacity_gb for gpu in potential_gpus)
    guarantee = total_gpu_units / tenant_count
    scenario.queues = tuple(
        QueueSpec(
            f"tenant-{index:02d}",
            "root",
            guaranteed=ResourceVector(guarantee),
            limit=ResourceVector(total_gpu_units, total_memory_gb),
        )
        for index in range(tenant_count)
    )
    for job in scenario.jobs:
        digest = hashlib.sha256(job.id.encode()).digest()
        job.queue_id = f"tenant-{int.from_bytes(digest[:8], 'big') % tenant_count:02d}"
    scenario.metadata["tenant_assignment"] = "synthetic_overlay"
    scenario.metadata["synthetic_tenant_count"] = tenant_count
    return scenario


def _trace_metadata(config: ExperimentConfig, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if str(config.workload.get("type")) != "trace":
        return {"trace_source": None, "trace_version": None}
    scenario = _scenario_for_seed(config, config.seeds[0])
    return {
        "trace_source": scenario.metadata.get("source_url"),
        "trace_version": scenario.metadata.get("trace_version"),
        "scenario_hash": runs[0]["scenario_hash"] if runs else None,
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_summary_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fields = (
        list(rows[0])
        if rows
        else ["scheduler", "metric", "runs", "mean", "stddev", "median", "p95"]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
