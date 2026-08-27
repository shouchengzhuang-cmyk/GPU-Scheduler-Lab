from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_scheduler_lab.allocation.allocator import FairShareScheduler
from gpu_scheduler_lab.experiments.manifest import git_sha, scenario_hash
from gpu_scheduler_lab.heterogeneous.config import (
    HeterogeneousStudyConfig,
    HeterogeneousStudyMode,
)
from gpu_scheduler_lab.heterogeneous.profile import EvidenceKind
from gpu_scheduler_lab.integrations import CONTRACT_V2_VERSION, import_mini_ai_cloud_export
from gpu_scheduler_lab.models.accelerator import AcceleratorVendor
from gpu_scheduler_lab.models.events import EventType
from gpu_scheduler_lab.models.job import JobStatus
from gpu_scheduler_lab.queues.hierarchy import QueueHierarchy
from gpu_scheduler_lab.scenario import Scenario, load_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.simulator.engine import SimulationResult, Simulator


@dataclass(frozen=True, slots=True)
class HeterogeneousStudyArtifacts:
    manifest: Path
    runs: Path
    report: Path


def run_heterogeneous_study(config_path: Path) -> HeterogeneousStudyArtifacts:
    config = HeterogeneousStudyConfig.load(config_path)
    scenario = load_scenario(config.scenario)
    inventory = _validate_typed_scenario(scenario)
    contract_check = _validate_v2_fixture(config.v2_contract_fixture)
    baseline_capacity = scenario.cluster.total_gpu_count
    runs: list[dict[str, Any]] = []
    variants: list[AcceleratorVendor | None] = [None, *config.outage_vendors]
    for outage_vendor in variants:
        variant = _with_vendor_outage(scenario, outage_vendor)
        for policy in config.route_policies:
            result = Simulator.from_scenario(
                variant,
                _scheduler_for(variant, policy),
            ).run()
            runs.append(
                _run_record(
                    variant,
                    result,
                    route_policy=policy,
                    outage_vendor=outage_vendor,
                    baseline_capacity=baseline_capacity,
                )
            )
    if any(run["cross_vendor_gang_violation_count"] for run in runs):
        raise AssertionError("heterogeneous study produced a cross-vendor gang placement")
    profiles = [profile.to_dict() for profile in config.performance_profiles]
    comparison_status = _performance_comparison_status(config)
    evidence = {
        "facts": [
            f"Scenario contains explicit typed inventory: {inventory}.",
            (
                "The configured Mini AI Cloud export passed the v2 adapter with "
                f"{contract_check['device_count']} typed devices."
            ),
            "Every observed gang placement used one accelerator vendor.",
        ],
        "assumptions": [
            "The discrete-event simulator models scheduling logic, not device execution.",
            (
                "Configured job duration and memory demand are study inputs, "
                "not hardware measurements."
            ),
            (
                "Quota, fair-share, reclaim, topology, and fragmentation use "
                "existing simulator semantics."
            ),
        ],
        "synthetic_variables": [
            f"Route policies: {', '.join(config.route_policies)}.",
            "Outage variants remove one configured vendor from schedulable capacity at time zero.",
            f"Scenario source: {config.scenario.as_posix()}.",
        ],
    }
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    artifacts = HeterogeneousStudyArtifacts(
        manifest=output / "manifest.json",
        runs=output / "runs.json",
        report=output / "report.md",
    )
    manifest = {
        "contract_version": "gpu-scheduler-lab.heterogeneous-study/v1",
        "name": config.name,
        "mode": config.mode.value,
        "evidence_kind": "SIMULATED",
        "git_sha": git_sha(config.source_path.parent),
        "scenario_hash": scenario_hash(scenario),
        "inventory": inventory,
        "v2_contract_check": contract_check,
        "route_policies": list(config.route_policies),
        "outage_vendors": [vendor.value for vendor in config.outage_vendors],
        "performance_profiles": profiles,
        "performance_comparison": comparison_status,
        "evidence": evidence,
        "real_hardware": {
            "nvidia": "REAL_HW_NOT_RUN",
            "huawei-ascend": "REAL_HW_NOT_RUN",
        },
    }
    _write_json(artifacts.manifest, manifest)
    _write_json(artifacts.runs, {"runs": runs})
    artifacts.report.write_text(
        _render_report(config, manifest, runs),
        encoding="utf-8",
        newline="\n",
    )
    return artifacts


def _validate_typed_scenario(scenario: Scenario) -> dict[str, int]:
    inventory: dict[str, int] = {}
    for device in scenario.cluster.gpus:
        if device.accelerator_metadata_inferred or device.vendor is AcceleratorVendor.UNKNOWN:
            raise ValueError("heterogeneous scenario devices must have explicit vendor and kind")
        inventory[device.vendor.value] = inventory.get(device.vendor.value, 0) + 1
    required = {AcceleratorVendor.NVIDIA.value, AcceleratorVendor.HUAWEI_ASCEND.value}
    if not required.issubset(inventory):
        raise ValueError("heterogeneous scenario must contain NVIDIA and Huawei Ascend inventory")
    if any(not job.accelerator_request_explicit for job in scenario.jobs):
        raise ValueError("heterogeneous scenario jobs must have explicit accelerator requests")
    return dict(sorted(inventory.items()))


def _validate_v2_fixture(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    imported = import_mini_ai_cloud_export(payload)
    if imported.metadata.get("contract_version") != CONTRACT_V2_VERSION:
        raise ValueError("heterogeneous study requires a Mini AI Cloud v2 export fixture")
    if any(device.accelerator_metadata_inferred for device in imported.cluster.gpus):
        raise ValueError("v2 export fixture must not contain inferred accelerator metadata")
    return {
        "contract_version": CONTRACT_V2_VERSION,
        "device_count": len(imported.cluster.gpus),
        "job_count": len(imported.jobs),
        "vendors": sorted({device.vendor.value for device in imported.cluster.gpus}),
    }


def _with_vendor_outage(scenario: Scenario, outage_vendor: AcceleratorVendor | None) -> Scenario:
    variant = scenario.clone()
    if outage_vendor is None:
        variant.metadata["vendor_outage"] = None
        return variant
    for node in variant.cluster.nodes:
        if any(device.vendor is outage_vendor for device in node.gpus):
            node.available = False
    variant.metadata["vendor_outage"] = outage_vendor.value
    return variant


def _scheduler_for(scenario: Scenario, policy: str) -> Scheduler:
    placement = create_scheduler(policy, scenario)
    if not scenario.queues:
        return placement
    return FairShareScheduler(
        QueueHierarchy(scenario.queues),
        scenario.accounting,
        placement=placement,
        historical=True,
        half_life=scenario.fairshare_half_life,
        borrowing=True,
        reclaim=True,
        name=policy,
    )


def _run_record(
    scenario: Scenario,
    result: SimulationResult,
    *,
    route_policy: str,
    outage_vendor: AcceleratorVendor | None,
    baseline_capacity: int,
) -> dict[str, Any]:
    placements: dict[str, int] = {}
    cross_vendor_gangs = 0
    placement_events = {
        EventType.JOB_START,
        EventType.JOB_RESUME,
        EventType.JOB_RESTART,
        EventType.ELASTIC_SCALE_UP,
    }
    for record in result.trace:
        if record.event not in placement_events or not record.gpu_ids:
            continue
        vendors = {
            scenario.cluster.gpu_by_id(device_id).vendor.value for device_id in record.gpu_ids
        }
        if len(vendors) > 1:
            cross_vendor_gangs += 1
        for vendor in vendors:
            placements[vendor] = placements.get(vendor, 0) + 1
    selected_metrics = {
        name: result.metrics[name]
        for name in (
            "average_gpu_utilization",
            "gpu_memory_utilization",
            "gpu_fragmentation_ratio",
            "average_waiting_time",
            "p95_waiting_time",
            "completion_rate",
            "sla_violation_rate",
            "jains_fairness_index",
            "queue_service_jains_index",
            "guarantee_satisfaction_variance",
            "rejected_job_count",
            "failed_placement_attempt_count",
        )
    }
    selected_metrics["reclaim_victim_count"] = sum(job.reclaim_victim_count for job in result.jobs)
    capacity = scenario.cluster.total_gpu_count
    return {
        "variant": "baseline" if outage_vendor is None else f"outage-{outage_vendor.value}",
        "outage_vendor": outage_vendor.value if outage_vendor is not None else None,
        "route_policy": route_policy,
        "schedulable_devices": capacity,
        "capacity_loss_fraction": (
            (baseline_capacity - capacity) / baseline_capacity if baseline_capacity else 0.0
        ),
        "placements_by_vendor": dict(sorted(placements.items())),
        "cross_vendor_gang_violation_count": cross_vendor_gangs,
        "vendor_restricted_jobs": sum(bool(job.allowed_vendors) for job in result.jobs),
        "job_status_counts": {
            status.value: sum(job.status is status for job in result.jobs) for status in JobStatus
        },
        "metrics": selected_metrics,
    }


def _performance_comparison_status(config: HeterogeneousStudyConfig) -> dict[str, str]:
    if config.mode is HeterogeneousStudyMode.CORRECTNESS:
        return {
            "status": "NOT_APPLICABLE",
            "reason": "Correctness mode does not compare hardware performance.",
        }
    kinds = {profile.source_kind for profile in config.performance_profiles}
    if kinds != {EvidenceKind.MEASURED}:
        return {
            "status": "NOT_PERMITTED",
            "reason": "All compared profiles must be MEASURED before ranking vendors.",
        }
    return {
        "status": "RAW_MEASURED_PROFILES_ONLY",
        "reason": "Measured values are reported without an automatic vendor winner claim.",
    }


def _render_report(
    config: HeterogeneousStudyConfig,
    manifest: dict[str, Any],
    runs: list[dict[str, Any]],
) -> str:
    evidence = manifest["evidence"]
    lines = [
        f"# {config.name}",
        "",
        f"Mode: `{config.mode.value}`. Evidence kind: `SIMULATED`.",
        "",
        "## Facts",
        "",
        *[f"- {value}" for value in evidence["facts"]],
        "",
        "## Assumptions",
        "",
        *[f"- {value}" for value in evidence["assumptions"]],
        "",
        "## Synthetic variables",
        "",
        *[f"- {value}" for value in evidence["synthetic_variables"]],
        "",
        "## Correctness results",
        "",
        (
            "| Variant | Policy | Capacity | Completion | Fragmentation | Fairness | "
            "Reclaims | Cross-vendor gangs |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        metrics = run["metrics"]
        lines.append(
            f"| {run['variant']} | {run['route_policy']} | {run['schedulable_devices']} | "
            f"{metrics['completion_rate']:.3f} | {metrics['gpu_fragmentation_ratio']:.3f} | "
            f"{metrics['queue_service_jains_index']:.3f} | "
            f"{metrics['reclaim_victim_count']} | "
            f"{run['cross_vendor_gang_violation_count']} |"
        )
    comparison = manifest["performance_comparison"]
    lines.extend(
        [
            "",
            "## Performance evidence boundary",
            "",
            f"Status: `{comparison['status']}`. {comparison['reason']}",
            "",
        ]
    )
    profiles = manifest["performance_profiles"]
    if profiles:
        lines.extend(
            [
                (
                    "| Source kind | Source ID | Model variant | TTFT ms | TPOT ms | "
                    "Throughput tokens/s | Power W | Cost/hour |"
                ),
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for profile in profiles:
            lines.append(
                f"| {profile['source_kind']} | {profile['source_id']} | "
                f"{profile['model_variant']} | {profile['ttft_ms']} | "
                f"{profile['tpot_ms']} | {profile['throughput_tokens_s']} | "
                f"{profile['power_watts']} | {profile['cost_per_hour']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Real hardware boundary",
            "",
            "- NVIDIA: `REAL_HW_NOT_RUN`.",
            "- Huawei Ascend: `REAL_HW_NOT_RUN`.",
            (
                "- These results do not validate CUDA, CANN, Kubernetes device plugins, "
                "or production throughput."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
