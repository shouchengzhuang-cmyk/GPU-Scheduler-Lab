from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from gpu_scheduler_lab.heterogeneous.profile import PerformanceProfile
from gpu_scheduler_lab.models.accelerator import AcceleratorVendor


class HeterogeneousStudyMode(StrEnum):
    CORRECTNESS = "correctness"
    CALIBRATED = "calibrated"


@dataclass(frozen=True, slots=True)
class HeterogeneousStudyConfig:
    name: str
    mode: HeterogeneousStudyMode
    scenario: Path
    v2_contract_fixture: Path
    route_policies: tuple[str, ...]
    outage_vendors: tuple[AcceleratorVendor, ...]
    performance_profiles: tuple[PerformanceProfile, ...]
    output_directory: Path
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> HeterogeneousStudyConfig:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("heterogeneous study config root must be a mapping")
        study = raw.get("study", {})
        output = raw.get("output", {})
        if not isinstance(study, dict) or not isinstance(output, dict):
            raise ValueError("study and output must be mappings")
        name = str(study.get("name", "")).strip()
        if not name:
            raise ValueError("study.name must not be empty")
        mode = HeterogeneousStudyMode(str(study.get("mode", "correctness")))
        scenario = _required_path(raw, "scenario")
        v2_fixture = _required_path(raw, "v2_contract_fixture")
        policies_raw = raw.get("route_policies", [])
        if not isinstance(policies_raw, list) or not policies_raw:
            raise ValueError("route_policies must be a non-empty list")
        policies = tuple(str(value) for value in policies_raw)
        supported_policies = {"prefer-nvidia", "prefer-ascend"}
        if any(policy not in supported_policies for policy in policies):
            raise ValueError("route_policies only supports prefer-nvidia and prefer-ascend")
        if len(set(policies)) != len(policies):
            raise ValueError("route_policies must not contain duplicates")
        outages_raw = raw.get("outage_vendors", [])
        if not isinstance(outages_raw, list):
            raise ValueError("outage_vendors must be a list")
        outages = tuple(AcceleratorVendor(str(value)) for value in outages_raw)
        if AcceleratorVendor.UNKNOWN in outages:
            raise ValueError("outage_vendors must be explicit")
        if len(set(outages)) != len(outages):
            raise ValueError("outage_vendors must not contain duplicates")
        profiles_raw = raw.get("performance_profiles", [])
        if not isinstance(profiles_raw, list):
            raise ValueError("performance_profiles must be a list")
        profiles = tuple(PerformanceProfile.from_dict(value) for value in profiles_raw)
        if mode is HeterogeneousStudyMode.CORRECTNESS and profiles:
            raise ValueError("correctness mode must not include performance_profiles")
        if mode is HeterogeneousStudyMode.CALIBRATED and not profiles:
            raise ValueError("calibrated mode requires performance_profiles")
        directory = Path(str(output.get("directory", f"experiment-results/{name}")))
        return cls(
            name=name,
            mode=mode,
            scenario=_resolve(scenario),
            v2_contract_fixture=_resolve(v2_fixture),
            route_policies=policies,
            outage_vendors=outages,
            performance_profiles=profiles,
            output_directory=_resolve(directory),
            source_path=path,
        )


def _required_path(raw: dict[object, object], name: str) -> Path:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    return Path(value)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path
