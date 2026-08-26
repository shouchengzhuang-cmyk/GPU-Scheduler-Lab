from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml

from gpu_scheduler_lab.allocation.allocator import FairShareScheduler
from gpu_scheduler_lab.elastic.work import ElasticSpec
from gpu_scheduler_lab.experiments.manifest import git_sha
from gpu_scheduler_lab.fleet.events import FleetEvent, FleetEventType
from gpu_scheduler_lab.models.topology import TopologyMode
from gpu_scheduler_lab.queues.hierarchy import QueueHierarchy
from gpu_scheduler_lab.queues.model import QueueSpec, ResourceVector
from gpu_scheduler_lab.scenario import Scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.simulator.engine import Simulator
from gpu_scheduler_lab.study.config import Policy, StudyConfig
from gpu_scheduler_lab.workload import GeneratorConfig, generate_scenario

Scalar = str | int | float | bool


@dataclass(frozen=True, slots=True)
class ScenarioTemplate:
    id: str
    generator: dict[str, Any]
    controls: dict[str, Scalar]
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StudyVariant:
    id: str
    variables: tuple[tuple[str, Scalar], ...]
    ablation: str | None = None


@dataclass(frozen=True, slots=True)
class StudyRunPlan:
    run_id: str
    variant_id: str
    policy_id: str
    scheduler_name: str
    seed: int
    effective_seed: int
    replication: int
    variables: tuple[tuple[str, Scalar], ...]
    ablation: str | None
    config_sha256: str


@dataclass(frozen=True, slots=True)
class StudyArtifacts:
    output_directory: Path
    manifest: Path
    summary_json: Path
    summary_csv: Path
    run_count: int
    resumed_count: int


class StudyRunError(RuntimeError):
    pass


RunExecutor = Callable[[StudyConfig, ScenarioTemplate, StudyRunPlan], dict[str, float]]


def run_study(
    config_path: Path,
    *,
    executor: RunExecutor | None = None,
) -> StudyArtifacts:
    config = StudyConfig.load(config_path)
    template = load_scenario_template(config.scenario_path)
    revision = git_sha(config.source_path.parent) or "unknown"
    config_sha256 = study_config_hash(config, template, revision)
    plans = build_run_plan(config, template, config_sha256)
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    execute = executor or simulate_plan
    completed: list[dict[str, Any]] = []
    resumed_count = 0
    for plan in plans:
        cached = _load_completed_run(output, plan) if config.resume else None
        if cached is not None:
            completed.append(cached)
            resumed_count += 1
            continue
        run_directory = output / "runs" / plan.run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        for _ in range(config.warmup_runs):
            execute(config, template, plan)
        errors: list[str] = []
        metrics: dict[str, float] | None = None
        for attempt in range(config.max_retries + 1):
            try:
                metrics = execute(config, template, plan)
                break
            except Exception as exc:  # noqa: BLE001 - retry boundary records any failed run.
                error = _safe_error(exc)
                errors.append(error)
                _write_json(
                    run_directory / "attempts" / f"{attempt + 1:02d}.json",
                    {"attempt": attempt + 1, "error": error},
                )
        if metrics is None:
            _write_json(
                run_directory / "manifest.json",
                _run_manifest(plan, revision, status="failed", attempts=len(errors)),
            )
            raise StudyRunError(
                f"run {plan.run_id} failed after {len(errors)} attempts: {errors[-1]}"
            )
        record = _completed_record(plan, metrics)
        _write_json(run_directory / "result.json", record)
        _write_json(
            run_directory / "manifest.json",
            _run_manifest(plan, revision, status="complete", attempts=len(errors) + 1),
        )
        completed.append(record)

    completed.sort(key=_record_sort_key)
    summary = aggregate_study_runs(completed)
    manifest_path = output / "manifest.json"
    summary_json = output / "summary.json"
    summary_csv = output / "summary.csv"
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0.0",
            "study_id": config.id,
            "git_sha": revision,
            "config_sha256": config_sha256,
            "run_count": len(completed),
            "policy_ids": [policy.id for policy in config.policies],
            "seeds": sorted(config.seeds),
            "replications_per_seed": config.replications_per_seed,
            "warmup_runs": config.warmup_runs,
            "grid_mode": config.grid_mode,
            "ablations": list(config.ablations),
        },
    )
    _write_json(summary_json, {"schema_version": "1.0.0", "summary": summary})
    _write_summary_csv(summary_csv, summary)
    return StudyArtifacts(
        output_directory=output,
        manifest=manifest_path,
        summary_json=summary_json,
        summary_csv=summary_csv,
        run_count=len(completed),
        resumed_count=resumed_count,
    )


def load_scenario_template(path: Path) -> ScenarioTemplate:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("scenario"), dict):
        raise ValueError("canonical scenario must contain a scenario mapping")
    scenario = dict(raw["scenario"])
    generator = scenario.get("generator")
    controls = scenario.get("controls")
    if not isinstance(generator, dict) or not isinstance(controls, dict):
        raise ValueError("canonical scenario generator and controls must be mappings")
    scalar_controls: dict[str, Scalar] = {}
    for key, value in controls.items():
        if not isinstance(value, str | int | float | bool):
            raise ValueError(f"canonical scenario control {key!r} must be scalar")
        scalar_controls[str(key)] = value
    scenario_id = scenario.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("canonical scenario id must be a non-empty string")
    return ScenarioTemplate(
        id=scenario_id,
        generator=dict(generator),
        controls=scalar_controls,
        raw=dict(raw),
    )


def study_config_hash(config: StudyConfig, template: ScenarioTemplate, revision: str) -> str:
    payload = {
        "schema_version": config.schema_version,
        "study_id": config.id,
        "revision": revision,
        "policies": [
            {"id": policy.id, "scheduler": policy.scheduler, "mechanisms": policy.mechanisms}
            for policy in config.policies
        ],
        "metrics": [
            {"id": metric.id, "source_metric": metric.source_metric} for metric in config.metrics
        ],
        "variables": [
            {"id": variable.id, "parameter": variable.parameter, "values": variable.values}
            for variable in config.variables
        ],
        "seeds": sorted(config.seeds),
        "grid_mode": config.grid_mode,
        "warmup_runs": config.warmup_runs,
        "replications_per_seed": config.replications_per_seed,
        "ablations": config.ablations,
        "scenario": template.raw,
    }
    return _hash_payload(payload)


def build_run_plan(
    config: StudyConfig,
    template: ScenarioTemplate,
    config_sha256: str,
) -> tuple[StudyRunPlan, ...]:
    variants = build_variants(config, template)
    plans: list[StudyRunPlan] = []
    for variant in variants:
        for policy in config.policies:
            for seed in sorted(config.seeds):
                for replication in range(config.replications_per_seed):
                    effective_seed = seed + replication * 1_000_003
                    identity = {
                        "config_sha256": config_sha256,
                        "variant_id": variant.id,
                        "policy_id": policy.id,
                        "seed": seed,
                        "effective_seed": effective_seed,
                        "replication": replication,
                        "variables": variant.variables,
                        "ablation": variant.ablation,
                    }
                    plans.append(
                        StudyRunPlan(
                            run_id=_hash_payload(identity)[:20],
                            variant_id=variant.id,
                            policy_id=policy.id,
                            scheduler_name=policy.scheduler,
                            seed=seed,
                            effective_seed=effective_seed,
                            replication=replication,
                            variables=variant.variables,
                            ablation=variant.ablation,
                            config_sha256=config_sha256,
                        )
                    )
    return tuple(plans)


def build_variants(
    config: StudyConfig,
    template: ScenarioTemplate,
) -> tuple[StudyVariant, ...]:
    baseline = {
        variable.id: _template_parameter(template, variable.parameter)
        for variable in config.variables
    }
    variants: list[StudyVariant] = []
    if config.grid_mode == "cartesian":
        names = tuple(variable.id for variable in config.variables)
        for combination in itertools.product(*(variable.values for variable in config.variables)):
            variables = tuple(zip(names, combination, strict=True))
            variants.append(
                StudyVariant(id="grid-" + _hash_payload(variables)[:12], variables=variables)
            )
    else:
        baseline_tuple = tuple(
            (variable.id, baseline[variable.id]) for variable in config.variables
        )
        variants.append(StudyVariant(id="baseline", variables=baseline_tuple))
        for variable in config.variables:
            for value in variable.values:
                if value == baseline[variable.id]:
                    continue
                sensitivity_values = dict(baseline)
                sensitivity_values[variable.id] = value
                variants.append(
                    StudyVariant(
                        id=f"sensitivity-{variable.id}-{_slug(value)}",
                        variables=tuple(
                            (item.id, sensitivity_values[item.id]) for item in config.variables
                        ),
                    )
                )
    baseline_tuple = tuple((variable.id, baseline[variable.id]) for variable in config.variables)
    variants.extend(
        StudyVariant(
            id=f"ablation-{ablation}",
            variables=baseline_tuple,
            ablation=ablation,
        )
        for ablation in config.ablations
    )
    unique: dict[str, StudyVariant] = {variant.id: variant for variant in variants}
    return tuple(unique[variant_id] for variant_id in sorted(unique))


def simulate_plan(
    config: StudyConfig,
    template: ScenarioTemplate,
    plan: StudyRunPlan,
) -> dict[str, float]:
    scenario = scenario_for_plan(template, plan)
    policy = config.require_policy(plan.policy_id)
    scheduler, _mechanisms = scheduler_for_plan(policy, scenario, plan.ablation)
    result = Simulator(
        scenario.cluster,
        scenario.jobs,
        scheduler,
        scenario=scenario,
    ).run()
    extracted: dict[str, float] = {}
    for metric in config.metrics:
        value = _extract_metric(result.metrics, metric.source_metric)
        if value is not None:
            extracted[metric.id] = value
    return extracted


def scenario_for_plan(template: ScenarioTemplate, plan: StudyRunPlan) -> Scenario:
    values = dict(plan.variables)
    generator = template.generator
    config = GeneratorConfig(
        job_count=int(generator.get("jobs", 100)),
        node_count=int(generator.get("nodes", 8)),
        gpus_per_node=int(generator.get("gpus_per_node", 8)),
        arrival_rate=float(values["workload-intensity"]),
        median_duration=float(generator.get("median_duration", 60.0)),
        duration_distribution=str(generator.get("duration_distribution", "lognormal")),
        training_ratio=float(generator.get("training_ratio", 0.35)),
        gang_probability=float(generator.get("gang_probability", 0.35)),
        sla_probability=float(generator.get("sla_probability", 0.5)),
        seed=plan.effective_seed,
        profile=str(generator.get("profile", "mixed")),
    )
    scenario = generate_scenario(config)
    _apply_gpu_heterogeneity(scenario, str(values["gpu-heterogeneity"]))
    _apply_topology_strictness(scenario, str(values["topology-strictness"]))
    recovery_cost = float(values["checkpoint-restart-cost"])
    for job in scenario.jobs:
        job.checkpoint_cost = recovery_cost
        job.restart_cost = recovery_cost
    _apply_elastic_jobs(scenario, float(generator.get("elastic_ratio", 0.2)))
    _apply_tenant_queues(scenario, int(generator.get("tenant_count", 4)))
    _apply_revocable_capacity(scenario, float(values["revocable-capacity-ratio"]), config)
    scenario.metadata.update(
        {
            "study_variant": plan.variant_id,
            "study_policy": plan.policy_id,
            "study_ablation": plan.ablation,
            "study_variables": values,
        }
    )
    return scenario


def scheduler_for_plan(
    policy: Policy,
    scenario: Scenario,
    ablation: str | None,
) -> tuple[Scheduler, dict[str, bool]]:
    topology = policy.id in {"topology-aware", "historical-drf", "fairshare-reclaim"}
    history = policy.id == "historical-drf"
    reclaim = policy.id in {"historical-drf", "fairshare-reclaim"}
    elastic = policy.id in {"historical-drf", "fairshare-reclaim"}
    if ablation == "topology":
        topology = False
    elif ablation == "history":
        history = False
    elif ablation == "reclaim":
        reclaim = False
    elif ablation == "elastic":
        elastic = False
    mechanisms = {
        "topology": topology,
        "history": history,
        "reclaim": reclaim,
        "elastic": elastic,
    }
    if policy.id == "binpack":
        return create_scheduler("binpack", scenario), mechanisms
    if policy.id == "topology-aware":
        scheduler_name = "topology" if topology else "binpack"
        return create_scheduler(scheduler_name, scenario), mechanisms
    placement = create_scheduler("topology" if topology else "binpack", scenario)
    return (
        FairShareScheduler(
            QueueHierarchy(scenario.queues),
            scenario.accounting,
            placement=placement,
            historical=history,
            half_life=scenario.fairshare_half_life,
            borrowing=True,
            reclaim=reclaim,
            elastic=elastic,
            name=policy.id,
        ),
        mechanisms,
    )


def aggregate_study_runs(runs: list[dict[str, Any]]) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for run in runs:
        metrics = run.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("completed study run metrics must be a mapping")
        policy_id = str(run["policy_id"])
        variant_id = str(run["variant_id"])
        for metric_id, value in metrics.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    grouped[(policy_id, variant_id, str(metric_id))].append(numeric)
    rows: list[dict[str, str | int | float]] = []
    for (policy_id, variant_id, metric_id), samples in sorted(grouped.items()):
        rows.append(
            {
                "policy_id": policy_id,
                "variant_id": variant_id,
                "metric_id": metric_id,
                "samples": len(samples),
                "mean": mean(samples),
                "stddev": pstdev(samples),
            }
        )
    return rows


def _apply_gpu_heterogeneity(scenario: Scenario, mode: str) -> None:
    if mode == "balanced":
        return
    if mode not in {"homogeneous", "skewed"}:
        raise ValueError(f"unknown gpu heterogeneity mode: {mode}")
    for index, node in enumerate(scenario.cluster.nodes):
        if mode == "homogeneous":
            model, memory = "A100-40GB", 40.0
        elif index % 4 == 3:
            model, memory = "A100-80GB", 80.0
        else:
            model, memory = "A10", 24.0
        for gpu in node.gpus:
            gpu.model = model
            gpu.memory_capacity_gb = memory


def _apply_topology_strictness(scenario: Scenario, mode: str) -> None:
    mapping = {
        "none": TopologyMode.NONE,
        "rack": TopologyMode.REQUIRE_SAME_RACK,
        "node": TopologyMode.REQUIRE_SAME_NODE,
    }
    try:
        topology_mode = mapping[mode]
    except KeyError as exc:
        raise ValueError(f"unknown topology strictness: {mode}") from exc
    for job in scenario.jobs:
        if job.gpu_count > 1:
            job.topology_mode = topology_mode


def _apply_elastic_jobs(scenario: Scenario, ratio: float) -> None:
    if not 0 <= ratio <= 1:
        raise ValueError("elastic_ratio must be between zero and one")
    eligible = [job for job in scenario.jobs if job.gpu_count > 1]
    elastic_count = round(len(eligible) * ratio)
    total_gpus = scenario.cluster.total_gpu_count
    for job in eligible[:elastic_count]:
        maximum = min(total_gpus, max(job.gpu_count, job.gpu_count * 2))
        job.elastic = ElasticSpec(
            min_replicas=max(1, job.gpu_count // 2),
            preferred_replicas=job.gpu_count,
            max_replicas=maximum,
        )
        job.requested_replicas = job.elastic.preferred_replicas


def _apply_tenant_queues(scenario: Scenario, tenant_count: int) -> None:
    if tenant_count <= 0:
        raise ValueError("tenant_count must be positive")
    capacity = float(scenario.cluster.total_gpu_count)
    guarantee = capacity / tenant_count
    scenario.queues = tuple(
        QueueSpec(
            id=f"tenant-{index:02d}",
            parent="root",
            guaranteed=ResourceVector(gpu_units=guarantee),
            limit=ResourceVector(gpu_units=capacity),
        )
        for index in range(tenant_count)
    )
    for index, job in enumerate(scenario.jobs):
        job.queue_id = f"tenant-{index % tenant_count:02d}"


def _apply_revocable_capacity(
    scenario: Scenario,
    ratio: float,
    generator: GeneratorConfig,
) -> None:
    if not 0 <= ratio <= 1:
        raise ValueError("revocable capacity ratio must be between zero and one")
    node_count = round(len(scenario.cluster.nodes) * ratio)
    selected = scenario.cluster.nodes[:node_count]
    for node in selected:
        node.revocable = True
    revoke_time = generator.median_duration * 1.5
    return_time = generator.median_duration * 3.0
    scenario.fleet_events = tuple(
        event
        for node in selected
        for event in (
            FleetEvent(revoke_time, FleetEventType.CAPACITY_REVOKE, node.id),
            FleetEvent(return_time, FleetEventType.CAPACITY_RETURN, node.id),
        )
    )


def _extract_metric(metrics: dict[str, Any], source: str) -> float | None:
    if source == "queue_metrics.*.guaranteed_share_satisfaction":
        queues = metrics.get("queue_metrics")
        if not isinstance(queues, dict):
            return None
        samples = [
            float(value["guaranteed_share_satisfaction"])
            for value in queues.values()
            if isinstance(value, dict)
            and isinstance(value.get("guaranteed_share_satisfaction"), int | float)
        ]
        return mean(samples) if samples else None
    value = metrics.get(source)
    if isinstance(value, int | float) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _template_parameter(template: ScenarioTemplate, parameter: str) -> Scalar:
    section, separator, key = parameter.partition(".")
    if not separator or section not in {"generator", "controls"}:
        raise ValueError(f"unsupported study parameter: {parameter}")
    values: dict[str, Any] = template.generator if section == "generator" else template.controls
    value = values.get(key)
    if not isinstance(value, str | int | float | bool):
        raise ValueError(f"study parameter is missing or non-scalar: {parameter}")
    return value


def _completed_record(plan: StudyRunPlan, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "variant_id": plan.variant_id,
        "policy_id": plan.policy_id,
        "scheduler_name": plan.scheduler_name,
        "seed": plan.seed,
        "effective_seed": plan.effective_seed,
        "replication": plan.replication,
        "variables": dict(plan.variables),
        "ablation": plan.ablation,
        "metrics": dict(sorted(metrics.items())),
    }


def _run_manifest(
    plan: StudyRunPlan,
    revision: str,
    *,
    status: str,
    attempts: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "run_id": plan.run_id,
        "status": status,
        "git_sha": revision,
        "config_sha256": plan.config_sha256,
        "variant_id": plan.variant_id,
        "policy_id": plan.policy_id,
        "scheduler_name": plan.scheduler_name,
        "seed": plan.seed,
        "effective_seed": plan.effective_seed,
        "replication": plan.replication,
        "variables": dict(plan.variables),
        "ablation": plan.ablation,
        "attempts": attempts,
    }


def _load_completed_run(output: Path, plan: StudyRunPlan) -> dict[str, Any] | None:
    run_directory = output / "runs" / plan.run_id
    manifest_path = run_directory / "manifest.json"
    result_path = run_directory / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("run_id") != plan.run_id
        or manifest.get("config_sha256") != plan.config_sha256
        or not isinstance(result, dict)
        or result.get("run_id") != plan.run_id
        or not isinstance(result.get("metrics"), dict)
    ):
        return None
    return result


def _record_sort_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record["variant_id"]),
        str(record["policy_id"]),
        int(record["seed"]),
        int(record["replication"]),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_summary_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fields = ["policy_id", "variant_id", "metric_id", "samples", "mean", "stddev"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _slug(value: Scalar) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return normalized or "value"


def _safe_error(exc: Exception) -> str:
    value = f"{type(exc).__name__}: {exc}"
    return re.sub(r"(?i)(token|password|secret)=\S+", r"\1=[REDACTED]", value)[:2000]
