from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gpu_scheduler_lab.cli import main
from gpu_scheduler_lab.integrations import import_mini_ai_cloud_export
from gpu_scheduler_lab.scenario import load_scenario
from gpu_scheduler_lab.workload import GeneratorConfig, generate_scenario


def test_seeded_workload_is_reproducible() -> None:
    config = GeneratorConfig(job_count=30, seed=42, profile="mixed")

    first = generate_scenario(config)
    second = generate_scenario(config)

    assert [job.clone() for job in first.jobs] == [job.clone() for job in second.jobs]
    assert first.metadata == second.metadata


def test_profiles_generate_requested_job_count() -> None:
    for profile in ("mixed", "fragmentation", "burst", "topology", "backfill"):
        scenario = generate_scenario(GeneratorConfig(job_count=17, seed=7, profile=profile))
        assert len(scenario.jobs) == 17


def test_explicit_default_shaped_priority_weights_override_burst_profile() -> None:
    default_burst = GeneratorConfig(profile="burst")
    explicit = GeneratorConfig(profile="burst", priority_weights=(20, 50, 25, 5))

    assert default_burst.resolved_priority_weights == (15, 40, 35, 10)
    assert explicit.resolved_priority_weights == (20, 50, 25, 5)


def test_custom_workload_distributions_are_honored() -> None:
    scenario = generate_scenario(
        GeneratorConfig(
            job_count=12,
            training_ratio=0,
            duration_distribution="fixed",
            median_duration=20,
            gpu_count_distribution=((2, 1.0),),
            gpu_memory_distribution=((18.0, 1.0),),
            priority_weights=(0, 0, 0, 1),
            seed=9,
        )
    )

    assert {job.duration for job in scenario.jobs} == {9.0}
    assert {job.gpu_count for job in scenario.jobs} == {2}
    assert {job.gpu_memory_gb for job in scenario.jobs} == {18.0}
    assert {job.priority.name for job in scenario.jobs} == {"CRITICAL"}


def test_custom_gpu_count_distribution_preserves_cross_node_requests() -> None:
    scenario = generate_scenario(
        GeneratorConfig(
            job_count=4,
            node_count=8,
            gpus_per_node=8,
            training_ratio=1,
            gpu_count_distribution=((32, 1.0),),
            seed=10,
        )
    )

    assert {job.gpu_count for job in scenario.jobs} == {32}


def test_custom_gpu_count_distribution_rejects_oversized_requests() -> None:
    with pytest.raises(ValueError, match="more GPUs than the cluster contains"):
        GeneratorConfig(
            node_count=8,
            gpus_per_node=8,
            gpu_count_distribution=((65, 1.0),),
        )


def test_mini_ai_cloud_contract_maps_inventory_and_tasks() -> None:
    payload = {
        "contract_version": "mini-ai-cloud.gpu-scheduler-lab/v1",
        "workers": [
            {
                "id": "worker-a",
                "gpu_devices": [
                    {"device_uuid": "healthy", "memory_total_mb": 81920, "health": "healthy"},
                    {"device_uuid": "bad", "memory_total_mb": 81920, "health": "unhealthy"},
                ],
            }
        ],
        "tasks": [
            {
                "id": "gpu-task",
                "queued_at": "2026-08-25T00:00:05Z",
                "duration_seconds": 30,
                "gpu_count": 1,
                "gpu_memory_mb": 40960,
                "priority": 95,
                "project_id": "p1",
                "workload_type": "model_inference",
            },
            {"id": "cpu-task", "gpu_count": 0},
        ],
    }

    scenario = import_mini_ai_cloud_export(payload)

    assert scenario.cluster.total_gpu_count == 1
    assert scenario.jobs[0].gpu_memory_gb == 40
    assert scenario.jobs[0].priority.name == "CRITICAL"
    assert scenario.metadata["cpu_only_tasks_filtered"] == 1


def test_mini_ai_cloud_contract_requires_gpu_memory() -> None:
    payload = {
        "contract_version": "mini-ai-cloud.gpu-scheduler-lab/v1",
        "workers": [],
        "tasks": [{"id": "bad", "gpu_count": 1, "gpu_memory_mb": 0}],
    }

    try:
        import_mini_ai_cloud_export(payload)
    except ValueError as exc:
        assert "positive gpu_memory_mb" in str(exc)
    else:
        raise AssertionError("expected missing GPU memory to be rejected")


def test_cli_writes_parseable_json_and_csv(tmp_path: Path) -> None:
    output = tmp_path / "results"
    main(
        [
            "compare",
            "--scenario",
            "scenarios/demo.yaml",
            "--schedulers",
            "fifo,binpack",
            "--output-dir",
            str(output),
            "--no-charts",
        ]
    )

    with (output / "demo-compare.json").open(encoding="utf-8") as handle:
        assert len(json.load(handle)["results"]) == 2
    rows = (output / "demo-compare.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3


def test_import_cli_writes_valid_scenario(tmp_path: Path) -> None:
    output = tmp_path / "imported.yaml"
    main(
        [
            "import-mini-ai-cloud",
            "--input",
            "scenarios/mini_ai_cloud_demo.json",
            "--output",
            str(output),
        ]
    )

    assert isinstance(yaml.safe_load(output.read_text(encoding="utf-8")), dict)
    scenario = load_scenario(output)
    assert len(scenario.jobs) == 2
