from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FORMAL_POLICY_IDS = (
    "binpack",
    "topology-aware",
    "historical-drf",
    "fairshare-reclaim",
)
FORMAL_METRIC_IDS = (
    "average-gpu-utilization",
    "p95-wait",
    "completion-rate",
    "gpu-count-fragmentation",
    "gpu-memory-fragmentation",
    "guaranteed-share-satisfaction",
    "jain-service-quality-fairness",
    "preemption-overhead",
    "checkpoint-overhead",
    "restart-overhead",
    "average-topology-distance",
    "topology-violation",
    "sla-violation",
)
FORMAL_VARIABLE_IDS = (
    "workload-intensity",
    "gpu-heterogeneity",
    "topology-strictness",
    "checkpoint-restart-cost",
    "revocable-capacity-ratio",
)
IMPLEMENTED_SCHEDULERS = {
    "binpack",
    "topology",
    "historical-drf",
    "fairshare-reclaim",
}


@dataclass(frozen=True, slots=True)
class Policy:
    id: str
    scheduler: str
    description: str
    mechanisms: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Metric:
    id: str
    source_metric: str
    unit: str
    direction: str
    description: str


@dataclass(frozen=True, slots=True)
class StudyVariable:
    id: str
    parameter: str
    values: tuple[str | int | float | bool, ...]
    description: str


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    statement: str
    independent_variable_ids: tuple[str, ...]
    dependent_metric_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StudyConfig:
    schema_version: str
    id: str
    title: str
    research_question: str
    policies: tuple[Policy, ...]
    metrics: tuple[Metric, ...]
    variables: tuple[StudyVariable, ...]
    hypotheses: tuple[Hypothesis, ...]
    seeds: tuple[int, ...]
    scenario_path: Path
    output_directory: Path
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> StudyConfig:
        source_path = path.resolve()
        root = _load_mapping(source_path, "study config")
        _require_keys(
            root,
            {
                "schema_version",
                "study",
                "policy_registry",
                "scenario",
                "policies",
                "metrics",
                "variables",
                "hypotheses",
                "execution",
            },
            "study config",
        )
        schema_version = _required_string(root, "schema_version", "study config")
        if schema_version != "1.0.0":
            raise ValueError("study.schema_version must be 1.0.0")

        study = _required_mapping(root, "study", "study config")
        _require_keys(study, {"id", "title", "research_question"}, "study")
        study_id = _required_string(study, "id", "study")
        title = _required_string(study, "title", "study")
        research_question = _required_string(study, "research_question", "study")

        policy_registry = _resolve_reference(
            source_path.parent,
            _required_string(root, "policy_registry", "study config"),
        )
        policies = _load_policies(policy_registry, schema_version)
        selected_policy_ids = _string_tuple(root.get("policies"), "policies")
        _require_exact_ids("formal policy", selected_policy_ids, FORMAL_POLICY_IDS)
        registry_by_id = {policy.id: policy for policy in policies}
        selected_policies = tuple(registry_by_id[policy_id] for policy_id in selected_policy_ids)

        metrics = tuple(
            _parse_metric(item, index)
            for index, item in enumerate(_mapping_list(root.get("metrics"), "metrics"))
        )
        _require_exact_ids("formal metric", (item.id for item in metrics), FORMAL_METRIC_IDS)

        variables = tuple(
            _parse_variable(item, index)
            for index, item in enumerate(_mapping_list(root.get("variables"), "variables"))
        )
        _require_exact_ids("formal variable", (item.id for item in variables), FORMAL_VARIABLE_IDS)

        hypotheses = tuple(
            _parse_hypothesis(item, index)
            for index, item in enumerate(_mapping_list(root.get("hypotheses"), "hypotheses"))
        )
        _require_unique("hypothesis", (item.id for item in hypotheses))
        if not hypotheses:
            raise ValueError("hypotheses must not be empty")
        variable_ids = {item.id for item in variables}
        metric_ids = {item.id for item in metrics}
        for hypothesis in hypotheses:
            _require_known(
                f"hypothesis {hypothesis.id} independent variable",
                hypothesis.independent_variable_ids,
                variable_ids,
            )
            _require_known(
                f"hypothesis {hypothesis.id} dependent metric",
                hypothesis.dependent_metric_ids,
                metric_ids,
            )

        scenario_path = _resolve_reference(
            source_path.parent,
            _required_string(root, "scenario", "study config"),
        )
        _validate_scenario(scenario_path, schema_version)
        _validate_schema_document(source_path.parent / "schema.json", schema_version)
        for document in ("hypotheses.md", "metric-definitions.md"):
            if not (source_path.parent / document).is_file():
                raise ValueError(f"missing study document: {document}")

        execution = _required_mapping(root, "execution", "study config")
        _require_keys(execution, {"seeds", "output_directory"}, "execution")
        raw_seeds = execution.get("seeds")
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise ValueError("execution.seeds must be a non-empty list")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in raw_seeds):
            raise ValueError("execution.seeds must contain integers")
        seeds = tuple(raw_seeds)
        _require_unique("seed", (str(seed) for seed in seeds))
        output_directory = _resolve_reference(
            source_path.parent,
            _required_string(execution, "output_directory", "execution"),
            require_exists=False,
        )
        return cls(
            schema_version=schema_version,
            id=study_id,
            title=title,
            research_question=research_question,
            policies=selected_policies,
            metrics=metrics,
            variables=variables,
            hypotheses=hypotheses,
            seeds=seeds,
            scenario_path=scenario_path,
            output_directory=output_directory,
            source_path=source_path,
        )

    def require_policy(self, policy_id: str) -> Policy:
        for policy in self.policies:
            if policy.id == policy_id:
                return policy
        allowed = ", ".join(policy.id for policy in self.policies)
        raise ValueError(f"policy {policy_id!r} is not registered for study {self.id!r}: {allowed}")


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{label} root must be a mapping")
    return dict(raw)


def _required_mapping(payload: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label}.{key} must be a mapping")
    return dict(value)


def _required_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"{label} is missing keys: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{label} has unknown keys: {', '.join(extra)}")


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def _mapping_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must contain mappings")
    return [dict(item) for item in value]


def _parse_metric(item: dict[str, Any], index: int) -> Metric:
    label = f"metrics[{index}]"
    _require_keys(item, {"id", "source_metric", "unit", "direction", "description"}, label)
    direction = _required_string(item, "direction", label)
    if direction not in {"maximize", "minimize", "context"}:
        raise ValueError(f"{label}.direction must be maximize, minimize, or context")
    return Metric(
        id=_required_string(item, "id", label),
        source_metric=_required_string(item, "source_metric", label),
        unit=_required_string(item, "unit", label),
        direction=direction,
        description=_required_string(item, "description", label),
    )


def _parse_variable(item: dict[str, Any], index: int) -> StudyVariable:
    label = f"variables[{index}]"
    _require_keys(item, {"id", "parameter", "values", "description"}, label)
    values = item.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label}.values must be a non-empty list")
    if any(not isinstance(value, str | int | float | bool) for value in values):
        raise ValueError(f"{label}.values must contain scalar values")
    return StudyVariable(
        id=_required_string(item, "id", label),
        parameter=_required_string(item, "parameter", label),
        values=tuple(values),
        description=_required_string(item, "description", label),
    )


def _parse_hypothesis(item: dict[str, Any], index: int) -> Hypothesis:
    label = f"hypotheses[{index}]"
    _require_keys(
        item,
        {"id", "statement", "independent_variable_ids", "dependent_metric_ids"},
        label,
    )
    return Hypothesis(
        id=_required_string(item, "id", label),
        statement=_required_string(item, "statement", label),
        independent_variable_ids=_string_tuple(
            item.get("independent_variable_ids"), f"{label}.independent_variable_ids"
        ),
        dependent_metric_ids=_string_tuple(
            item.get("dependent_metric_ids"), f"{label}.dependent_metric_ids"
        ),
    )


def _load_policies(path: Path, schema_version: str) -> tuple[Policy, ...]:
    root = _load_mapping(path, "policy registry")
    _require_keys(root, {"schema_version", "policies"}, "policy registry")
    if _required_string(root, "schema_version", "policy registry") != schema_version:
        raise ValueError("policy registry schema_version does not match study")
    policies: list[Policy] = []
    for index, item in enumerate(_mapping_list(root.get("policies"), "policy registry.policies")):
        label = f"policy registry.policies[{index}]"
        _require_keys(
            item,
            {"id", "scheduler", "description", "mechanisms", "limitations"},
            label,
        )
        scheduler = _required_string(item, "scheduler", label)
        if scheduler not in IMPLEMENTED_SCHEDULERS:
            raise ValueError(f"{label}.scheduler is not implemented: {scheduler}")
        policies.append(
            Policy(
                id=_required_string(item, "id", label),
                scheduler=scheduler,
                description=_required_string(item, "description", label),
                mechanisms=_string_tuple(item.get("mechanisms"), f"{label}.mechanisms"),
                limitations=_string_tuple(item.get("limitations"), f"{label}.limitations"),
            )
        )
    _require_exact_ids("policy registry", (item.id for item in policies), FORMAL_POLICY_IDS)
    return tuple(policies)


def _require_exact_ids(label: str, values: Any, expected: tuple[str, ...]) -> None:
    identifiers = tuple(values)
    _require_unique(label, identifiers)
    missing = sorted(set(expected) - set(identifiers))
    extra = sorted(set(identifiers) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unknown {', '.join(extra)}")
        raise ValueError(f"{label} ids must match the frozen set: {'; '.join(details)}")


def _require_unique(label: str, values: Any) -> None:
    identifiers = tuple(values)
    duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {', '.join(duplicates)}")


def _require_known(label: str, values: tuple[str, ...], known: set[str]) -> None:
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"{label} ids are unknown: {', '.join(unknown)}")


def _resolve_reference(base: Path, raw: str, *, require_exists: bool = True) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if require_exists and not path.is_file():
        raise ValueError(f"referenced file does not exist: {path}")
    return path


def _validate_scenario(path: Path, schema_version: str) -> None:
    root = _load_mapping(path, "canonical scenario")
    _require_keys(root, {"schema_version", "scenario"}, "canonical scenario")
    if _required_string(root, "schema_version", "canonical scenario") != schema_version:
        raise ValueError("canonical scenario schema_version does not match study")
    scenario = _required_mapping(root, "scenario", "canonical scenario")
    _require_keys(
        scenario,
        {"id", "description", "generator", "controls", "limitations"},
        "canonical scenario.scenario",
    )
    _required_string(scenario, "id", "canonical scenario.scenario")
    _required_string(scenario, "description", "canonical scenario.scenario")
    _required_mapping(scenario, "generator", "canonical scenario.scenario")
    _required_mapping(scenario, "controls", "canonical scenario.scenario")
    _string_tuple(scenario.get("limitations"), "canonical scenario.scenario.limitations")


def _validate_schema_document(path: Path, schema_version: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing study schema: {path}")
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid study schema JSON: {exc}") from exc
    if not isinstance(schema, dict):
        raise ValueError("study schema root must be a mapping")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("study schema must declare properties")
    version = properties.get("schema_version")
    if not isinstance(version, dict) or version.get("const") != schema_version:
        raise ValueError("study schema version does not match study config")
