from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_scheduler_lab.heterogeneous import PerformanceProfile, run_heterogeneous_study


def _config(tmp_path: Path, *, mode: str) -> Path:
    profiles: list[dict[str, Any]] = []
    if mode == "calibrated":
        profiles = [
            {
                "source_kind": "SYNTHETIC",
                "source_id": "synthetic-nvidia",
                "model_variant": "demo@nvidia",
                "ttft_ms": 100,
                "tpot_ms": 10,
                "throughput_tokens_s": 100,
                "power_watts": 300,
                "cost_per_hour": 1,
            },
            {
                "source_kind": "SYNTHETIC",
                "source_id": "synthetic-ascend",
                "model_variant": "demo@ascend",
                "ttft_ms": 100,
                "tpot_ms": 10,
                "throughput_tokens_s": 100,
                "power_watts": 300,
                "cost_per_hour": 1,
            },
        ]
    payload = {
        "study": {"name": f"heterogeneous-{mode}", "mode": mode},
        "scenario": "scenarios/heterogeneous-dual-stack.yaml",
        "v2_contract_fixture": "tests/fixtures/mini_ai_cloud/v2-golden.json",
        "route_policies": ["prefer-nvidia", "prefer-ascend"],
        "outage_vendors": ["nvidia", "huawei-ascend"],
        "performance_profiles": profiles,
        "output": {"directory": str(tmp_path / "output")},
    }
    path = tmp_path / f"{mode}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_correctness_study_covers_vendor_outages_and_single_vendor_gangs(
    tmp_path: Path,
) -> None:
    artifacts = run_heterogeneous_study(_config(tmp_path, mode="correctness"))
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    payload = json.loads(artifacts.runs.read_text(encoding="utf-8"))
    report = artifacts.report.read_text(encoding="utf-8")

    assert manifest["evidence_kind"] == "SIMULATED"
    assert manifest["inventory"] == {"huawei-ascend": 4, "nvidia": 4}
    assert manifest["v2_contract_check"]["vendors"] == ["huawei-ascend", "nvidia"]
    assert manifest["performance_comparison"]["status"] == "NOT_APPLICABLE"
    assert manifest["real_hardware"] == {
        "nvidia": "REAL_HW_NOT_RUN",
        "huawei-ascend": "REAL_HW_NOT_RUN",
    }
    runs = payload["runs"]
    assert len(runs) == 6
    assert {run["variant"] for run in runs} == {
        "baseline",
        "outage-nvidia",
        "outage-huawei-ascend",
    }
    assert not any(run["cross_vendor_gang_violation_count"] for run in runs)
    assert {run["schedulable_devices"] for run in runs if run["variant"] != "baseline"} == {4}
    assert "## Facts" in report
    assert "## Assumptions" in report
    assert "## Synthetic variables" in report
    assert "REAL_HW_NOT_RUN" in report


def test_calibrated_study_keeps_synthetic_profiles_out_of_vendor_ranking(
    tmp_path: Path,
) -> None:
    artifacts = run_heterogeneous_study(_config(tmp_path, mode="calibrated"))
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    report = artifacts.report.read_text(encoding="utf-8")

    assert {profile["source_kind"] for profile in manifest["performance_profiles"]} == {"SYNTHETIC"}
    assert manifest["performance_comparison"]["status"] == "NOT_PERMITTED"
    assert "All compared profiles must be MEASURED" in report
    assert "faster than" not in report


def test_heterogeneous_runs_are_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="correctness")

    first = run_heterogeneous_study(config).runs.read_text(encoding="utf-8")
    second = run_heterogeneous_study(config).runs.read_text(encoding="utf-8")

    assert first == second


def test_performance_profile_requires_explicit_evidence_kind() -> None:
    with pytest.raises(ValueError, match="missing fields: source_kind"):
        PerformanceProfile.from_dict(
            {
                "source_id": "missing-kind",
                "model_variant": "demo",
                "ttft_ms": 1,
                "tpot_ms": 1,
                "throughput_tokens_s": 1,
                "power_watts": 0,
                "cost_per_hour": 0,
            }
        )
