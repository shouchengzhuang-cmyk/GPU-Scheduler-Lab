from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gpu_scheduler_lab.scenario import load_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.simulator.engine import Simulator

REQUIRED_INVARIANT_IDS = (
    "gpu-exclusive-ownership",
    "atomic-gang-allocation",
    "hierarchical-hard-limits",
    "no-borrow-ancestor-guarantees",
    "legal-entitlement-reclaim",
    "reclaim-target-within-guarantee",
    "required-topology-on-start-and-resize",
    "unavailable-capacity-excluded",
    "fleet-snapshots-not-unioned",
    "draining-capacity-remains-active",
    "fairshare-resorts-after-allocation",
    "stale-completion-generation-fenced",
)


@dataclass(frozen=True, slots=True)
class Invariant:
    id: str
    assertion: str
    tests: tuple[str, ...]
    golden_cases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    scenario: Path
    scheduler: str
    metrics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InvariantContract:
    path: Path
    invariants: tuple[Invariant, ...]
    cases_path: Path
    baseline_path: Path
    golden_cases: tuple[GoldenCase, ...]

    @classmethod
    def load(cls, path: Path) -> InvariantContract:
        resolved = path.resolve()
        repo_root = resolved.parent.parent
        raw = _load_mapping(resolved)
        if raw.get("version") != 1:
            raise ValueError("invariant contract version must be 1")
        raw_invariants = raw.get("invariants")
        if not isinstance(raw_invariants, list):
            raise ValueError("invariants must be a list")
        invariants = tuple(_parse_invariant(item) for item in raw_invariants)
        ids = tuple(item.id for item in invariants)
        if len(set(ids)) != len(ids):
            raise ValueError("invariant ids must be unique")
        if set(ids) != set(REQUIRED_INVARIANT_IDS) or len(ids) != len(REQUIRED_INVARIANT_IDS):
            raise ValueError("invariant contract must contain the required 12 ids exactly")

        golden = raw.get("golden")
        if not isinstance(golden, dict):
            raise ValueError("golden must be a mapping")
        cases_path = repo_root / _required_string(golden, "cases")
        baseline_path = repo_root / _required_string(golden, "baseline")
        cases = _load_golden_cases(cases_path, repo_root)
        case_ids = {case.id for case in cases}

        for invariant in invariants:
            if not invariant.tests:
                raise ValueError(f"invariant {invariant.id} must reference a semantic test")
            for test_ref in invariant.tests:
                test_path = repo_root / test_ref.split("::", 1)[0]
                if not test_path.is_file():
                    raise ValueError(f"invariant {invariant.id} references missing test {test_ref}")
            missing_cases = set(invariant.golden_cases) - case_ids
            if missing_cases:
                raise ValueError(
                    f"invariant {invariant.id} references unknown golden cases "
                    f"{sorted(missing_cases)}"
                )

        if not baseline_path.is_file():
            raise ValueError(f"golden baseline does not exist: {baseline_path}")
        return cls(resolved, invariants, cases_path, baseline_path, cases)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    return raw


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_tuple(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return tuple(value)


def _parse_invariant(raw: Any) -> Invariant:
    if not isinstance(raw, dict):
        raise ValueError("each invariant must be a mapping")
    return Invariant(
        id=_required_string(raw, "id"),
        assertion=_required_string(raw, "assertion"),
        tests=_string_tuple(raw, "tests"),
        golden_cases=_string_tuple(raw, "golden_cases"),
    )


def _load_golden_cases(path: Path, repo_root: Path) -> tuple[GoldenCase, ...]:
    raw = _load_mapping(path)
    if raw.get("version") != 1:
        raise ValueError("golden case version must be 1")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("golden cases must be a non-empty list")
    cases: list[GoldenCase] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            raise ValueError("each golden case must be a mapping")
        scenario = repo_root / _required_string(item, "scenario")
        if not scenario.is_file():
            raise ValueError(f"golden scenario does not exist: {scenario}")
        metrics = _string_tuple(item, "metrics")
        if not metrics:
            raise ValueError("golden case metrics must not be empty")
        cases.append(
            GoldenCase(
                id=_required_string(item, "id"),
                scenario=scenario,
                scheduler=_required_string(item, "scheduler"),
                metrics=metrics,
            )
        )
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("golden case ids must be unique")
    return tuple(cases)


def generate_baseline(contract: InvariantContract) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case in contract.golden_cases:
        scenario = load_scenario(case.scenario)
        result = Simulator.from_scenario(scenario, create_scheduler(case.scheduler, scenario)).run()
        missing = [metric for metric in case.metrics if metric not in result.metrics]
        if missing:
            raise ValueError(f"golden case {case.id} has unknown metrics {missing}")
        cases[case.id] = {
            "scenario": case.scenario.relative_to(contract.path.parent.parent).as_posix(),
            "scenario_sha256": hashlib.sha256(case.scenario.read_bytes()).hexdigest(),
            "scheduler": case.scheduler,
            "metrics": {metric: result.metrics[metric] for metric in case.metrics},
            "jobs": {
                job.id: {
                    "status": job.status.value,
                    "first_start_time": job.first_start_time,
                    "completion_time": job.completion_time,
                    "preemption_count": job.preemption_count,
                    "reclaim_victim_count": job.reclaim_victim_count,
                    "recovery_count": job.recovery_count,
                }
                for job in sorted(result.jobs, key=lambda item: item.id)
            },
        }
    return {"schema_version": 1, "cases": cases}


def render_baseline(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def validate_baseline(contract: InvariantContract) -> None:
    expected = contract.baseline_path.read_text(encoding="utf-8")
    actual = render_baseline(generate_baseline(contract))
    if actual != expected:
        raise ValueError(
            "golden baseline is stale; run `python scripts/update_invariant_baselines.py` "
            "to inspect the diff"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate scheduler invariant contract")
    parser.add_argument("--config", type=Path, default=Path("study/invariants.yaml"))
    args = parser.parse_args()
    contract = InvariantContract.load(args.config)
    validate_baseline(contract)
    print(
        f"Validated {len(contract.invariants)} invariants and "
        f"{len(contract.golden_cases)} golden cases."
    )


if __name__ == "__main__":
    main()
