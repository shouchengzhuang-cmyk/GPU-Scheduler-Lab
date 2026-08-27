from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from gpu_scheduler_lab.cli import build_parser
from gpu_scheduler_lab.study import StudyConfig
from gpu_scheduler_lab.study.report import generate_study_report, verify_hash_manifest
from gpu_scheduler_lab.study.runner import (
    ScenarioTemplate,
    StudyRunError,
    StudyRunPlan,
    build_run_plan,
    load_scenario_template,
    run_study,
    scenario_for_plan,
    scheduler_for_plan,
    simulate_plan,
    study_config_hash,
)

ROOT = Path(__file__).parents[1]
SMALL_CONFIG = ROOT / "study" / "study-small.yaml"


class _FlakyOncePerPlan:
    def __init__(self) -> None:
        self.failed = False

    def __call__(
        self,
        _config: StudyConfig,
        _template: ScenarioTemplate,
        plan: StudyRunPlan,
    ) -> dict[str, float]:
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected per-process first-attempt failure")
        return {"completion-rate": float(int(plan.run_id[:4], 16) % 100) / 100.0}


class _FailOnePlan:
    def __call__(
        self,
        _config: StudyConfig,
        _template: ScenarioTemplate,
        plan: StudyRunPlan,
    ) -> dict[str, float]:
        if (
            plan.variant_id == "baseline"
            and plan.policy_id == "binpack"
            and plan.seed == 3
            and plan.replication == 0
        ):
            raise RuntimeError("injected terminal plan failure")
        return {"completion-rate": float(int(plan.run_id[:4], 16) % 100) / 100.0}


def test_run_plan_is_stable_and_sorted_across_seeds() -> None:
    config = StudyConfig.load(SMALL_CONFIG)
    template = load_scenario_template(config.scenario_path)
    fingerprint = study_config_hash(config, template, "a" * 40)

    first = build_run_plan(config, template, fingerprint)
    second = build_run_plan(config, template, fingerprint)

    assert first == second
    assert [plan.run_id for plan in first] == [plan.run_id for plan in second]
    assert len({plan.run_id for plan in first}) == len(first)
    assert {plan.seed for plan in first} == {11, 29}
    assert [plan.variant_id for plan in first] == sorted(plan.variant_id for plan in first)


def test_same_config_seed_and_plan_produce_identical_metrics() -> None:
    config = StudyConfig.load(SMALL_CONFIG)
    template = load_scenario_template(config.scenario_path)
    fingerprint = study_config_hash(config, template, "b" * 40)
    plan = next(
        plan
        for plan in build_run_plan(config, template, fingerprint)
        if plan.variant_id == "baseline" and plan.policy_id == "binpack" and plan.seed == 11
    )

    assert simulate_plan(config, template, plan) == simulate_plan(config, template, plan)


def test_formal_fairshare_policies_do_not_receive_zero_memory_limits() -> None:
    config = StudyConfig.load(SMALL_CONFIG)
    template = load_scenario_template(config.scenario_path)
    fingerprint = study_config_hash(config, template, "d" * 40)
    plans = build_run_plan(config, template, fingerprint)

    for policy_id in ("historical-drf", "fairshare-reclaim"):
        plan = next(
            item
            for item in plans
            if item.variant_id == "baseline" and item.policy_id == policy_id and item.seed == 11
        )
        metrics = simulate_plan(config, template, plan)
        assert metrics["completion-rate"] > 0
        assert metrics["average-gpu-utilization"] > 0


def test_retry_resume_and_partial_result_recovery(tmp_path: Path) -> None:
    config_path = _temporary_overlay(tmp_path)
    calls: dict[str, int] = {}
    failed_once = False

    def flaky(
        _config: StudyConfig,
        _template: ScenarioTemplate,
        plan: StudyRunPlan,
    ) -> dict[str, float]:
        nonlocal failed_once
        run_id = plan.run_id
        calls[run_id] = calls.get(run_id, 0) + 1
        if not failed_once:
            failed_once = True
            raise RuntimeError("injected first-attempt failure")
        return {"completion-rate": float(int(run_id[:4], 16) % 100) / 100.0}

    first = run_study(config_path, executor=flaky)
    assert first.run_count > 1
    assert first.resumed_count == 0
    assert sum(calls.values()) == first.run_count + 1
    first_summary = first.summary_json.read_bytes()

    calls.clear()
    second = run_study(config_path, executor=flaky)
    assert second.run_count == first.run_count
    assert second.resumed_count == first.run_count
    assert calls == {}
    assert second.summary_json.read_bytes() == first_summary

    one_result = next((first.output_directory / "runs").glob("*/result.json"))
    one_result.unlink()
    calls.clear()
    recovered = run_study(config_path, executor=flaky)
    assert recovered.resumed_count == recovered.run_count - 1
    assert sum(calls.values()) == 1


def test_parallel_study_matches_serial_artifacts_and_hashes(tmp_path: Path) -> None:
    serial_config = _equivalent_overlay(tmp_path / "serial")
    parallel_config = _equivalent_overlay(tmp_path / "parallel")

    serial = run_study(serial_config, workers=1)
    parallel = run_study(parallel_config, workers=2)
    serial_report = generate_study_report(serial.output_directory)
    parallel_report = generate_study_report(parallel.output_directory)

    assert serial.run_count == parallel.run_count
    assert serial.resumed_count == parallel.resumed_count == 0
    assert _artifact_bytes(serial.output_directory) == _artifact_bytes(parallel.output_directory)
    assert serial_report.hashes.read_bytes() == parallel_report.hashes.read_bytes()
    assert verify_hash_manifest(serial.output_directory) == verify_hash_manifest(
        parallel.output_directory
    )


def test_parallel_retry_and_resume_keep_per_run_attempts(tmp_path: Path) -> None:
    config_path = _temporary_overlay(tmp_path)
    first = run_study(config_path, executor=_FlakyOncePerPlan(), workers=2)

    manifests = sorted((first.output_directory / "runs").glob("*/manifest.json"))
    assert len(manifests) == first.run_count
    assert all(json.loads(path.read_text(encoding="utf-8"))["attempts"] == 2 for path in manifests)
    assert all((path.parent / "attempts" / "01.json").is_file() for path in manifests)

    second = run_study(config_path, executor=_FlakyOncePerPlan(), workers=2)
    assert second.run_count == first.run_count
    assert second.resumed_count == first.run_count


def test_parallel_failure_persists_other_completed_runs_for_resume(tmp_path: Path) -> None:
    config_path = _equivalent_overlay(tmp_path / "failure")
    config = StudyConfig.load(config_path)
    template = load_scenario_template(config.scenario_path)
    expected_runs = len(build_run_plan(config, template, "f" * 64))

    with pytest.raises(StudyRunError, match="injected terminal plan failure"):
        run_study(config_path, executor=_FailOnePlan(), workers=2)

    run_directories = sorted((config.output_directory / "runs").iterdir())
    manifests = [
        json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for directory in run_directories
    ]
    assert len(run_directories) == expected_runs
    assert sum(manifest["status"] == "failed" for manifest in manifests) == 1
    assert sum(manifest["status"] == "complete" for manifest in manifests) == expected_runs - 1
    assert sum((directory / "result.json").is_file() for directory in run_directories) == (
        expected_runs - 1
    )

    recovered = run_study(config_path, executor=_FlakyOncePerPlan(), workers=2)
    assert recovered.run_count == expected_runs
    assert recovered.resumed_count == expected_runs - 1


def test_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="workers must be an integer >= 1"):
        run_study(SMALL_CONFIG, workers=0)


def test_study_run_workers_cli_is_explicit_and_serial_by_default() -> None:
    parser = build_parser()

    default = parser.parse_args(["study", "run", "--config", str(SMALL_CONFIG)])
    parallel = parser.parse_args(["study", "run", "--config", str(SMALL_CONFIG), "--workers", "4"])

    assert default.workers == 1
    assert parallel.workers == 4
    with pytest.raises(SystemExit):
        parser.parse_args(["study", "run", "--config", str(SMALL_CONFIG), "--workers", "0"])


def test_each_ablation_disables_only_its_named_mechanism() -> None:
    config = StudyConfig.load(SMALL_CONFIG)
    template = load_scenario_template(config.scenario_path)
    fingerprint = study_config_hash(config, template, "c" * 40)
    plan = next(
        plan
        for plan in build_run_plan(config, template, fingerprint)
        if plan.variant_id == "baseline" and plan.policy_id == "historical-drf"
    )
    scenario = scenario_for_plan(template, plan)
    policy = config.require_policy("historical-drf")
    _scheduler, baseline = scheduler_for_plan(policy, scenario, None)

    assert baseline == {"topology": True, "history": True, "reclaim": True, "elastic": True}
    for mechanism in baseline:
        _scheduler, ablated = scheduler_for_plan(policy, scenario, mechanism)
        assert ablated[mechanism] is False
        assert {key: value for key, value in ablated.items() if key != mechanism} == {
            key: value for key, value in baseline.items() if key != mechanism
        }


def test_every_completed_run_has_independent_manifest(tmp_path: Path) -> None:
    config_path = _temporary_overlay(tmp_path)

    def executor(
        _config: StudyConfig,
        _template: ScenarioTemplate,
        _plan: StudyRunPlan,
    ) -> dict[str, float]:
        return {"completion-rate": 1.0}

    artifacts = run_study(config_path, executor=executor)
    manifests = sorted((artifacts.output_directory / "runs").glob("*/manifest.json"))

    assert len(manifests) == artifacts.run_count
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "complete" for path in manifests
    )
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert isinstance(manifest["dirty_tree"], bool | type(None))
    assert artifacts.environment.is_file()
    assert artifacts.scenario_hashes.is_file()
    assert artifacts.runs_json.is_file()
    assert json.loads(artifacts.runs_json.read_text(encoding="utf-8"))["runs"]


def _temporary_overlay(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for document in ("schema.json", "hypotheses.md", "metric-definitions.md"):
        shutil.copyfile(ROOT / "study" / document, tmp_path / document)
    overlay = {
        "extends": str((ROOT / "study" / "study.yaml").resolve()),
        "scenario": str((ROOT / "study" / "scenarios" / "small.yaml").resolve()),
        "execution": {
            "seeds": [3, 7],
            "output_directory": str(tmp_path / "output"),
            "grid_mode": "one-at-a-time",
            "warmup_runs": 0,
            "replications_per_seed": 1,
            "max_retries": 1,
            "resume": True,
            "ablations": ["topology"],
        },
    }
    path = tmp_path / "study-overlay.yaml"
    path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    return path


def _equivalent_overlay(root: Path) -> Path:
    root.mkdir(parents=True)
    for document in ("schema.json", "hypotheses.md", "metric-definitions.md"):
        shutil.copyfile(ROOT / "study" / document, root / document)
    overlay = {
        "extends": str((ROOT / "study" / "study.yaml").resolve()),
        "scenario": str((ROOT / "study" / "scenarios" / "small.yaml").resolve()),
        "execution": {
            "seeds": [3, 7],
            "output_directory": "output",
            "grid_mode": "one-at-a-time",
            "warmup_runs": 0,
            "replications_per_seed": 1,
            "max_retries": 1,
            "resume": True,
            "ablations": ["topology"],
        },
    }
    path = root / "study-overlay.yaml"
    path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    return path


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
