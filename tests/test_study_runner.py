from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from gpu_scheduler_lab.study import StudyConfig
from gpu_scheduler_lab.study.runner import (
    ScenarioTemplate,
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
