from __future__ import annotations

import csv
import json
from pathlib import Path

from conftest import make_cluster

from gpu_scheduler_lab.experiments import run_experiment, scenario_hash
from gpu_scheduler_lab.experiments.aggregation import aggregate_runs
from gpu_scheduler_lab.experiments.runner import _apply_tenant_overlay
from gpu_scheduler_lab.fleet import FleetEvent, FleetEventType
from gpu_scheduler_lab.models import Job
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.scenario import Scenario, write_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.simulator.engine import Simulator


def test_scenario_hash_is_stable_and_sensitive() -> None:
    scenario = Scenario(make_cluster([[24]]), [Job("job", 0, 1, 1, 20)])

    assert scenario_hash(scenario) == scenario_hash(scenario.clone())
    changed = Scenario(make_cluster([[24]]), [Job("job", 0, 2, 1, 20)])
    assert scenario_hash(scenario) != scenario_hash(changed)


def test_multi_seed_aggregation_statistics() -> None:
    runs = [
        {"scheduler": "fifo", "metrics": {"average_gpu_utilization": value}}
        for value in (0.25, 0.5, 0.75)
    ]

    row = aggregate_runs(runs, ("average_gpu_utilization",))[0]

    assert row["mean"] == 0.5
    assert row["median"] == 0.5
    assert row["p95"] == 0.725
    assert float(row["stddev"]) > 0


def test_experiment_writes_manifest_runs_summary_and_chart(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    write_scenario(
        Scenario(make_cluster([[24, 24]]), [Job("job", 0, 2, 1, 20)]),
        scenario_path,
    )
    output = tmp_path / "output"
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: fixture",
                "workload:",
                "  type: trace",
                f"  scenario: {scenario_path.as_posix()}",
                "schedulers: [fifo, binpack]",
                "seeds: [1]",
                "output:",
                f"  directory: {output.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = run_experiment(config)

    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            artifacts.manifest,
            artifacts.runs,
            artifacts.summary_csv,
            artifacts.summary_json,
            artifacts.comparison,
        )
    )
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["name"] == "fixture"
    assert manifest["elastic_model_version"] == "ideal-linear-v1"
    assert len(manifest["queue_config_hash"]) == 64
    assert len(manifest["fleet_event_hash"]) == 64
    assert len(manifest["runs"]) == 2
    assert len({run["scenario_hash"] for run in manifest["runs"]}) == 1
    with artifacts.summary_csv.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))


def test_phase3_experiment_writes_strict_metrics_and_timelines(tmp_path: Path) -> None:
    output = tmp_path / "phase3"
    config = tmp_path / "phase3.yaml"
    scenario = Path(__file__).resolve().parents[1] / "scenarios/multi-tenant-borrow-reclaim.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: phase3-fixture",
                "workload:",
                "  type: scenario",
                f"  scenario: {scenario.as_posix()}",
                "allocation_policy:",
                "  type: historical-drf",
                "  half_life: 20",
                "placement_scheduler: {type: topology}",
                "queue_policy: {borrowing: true, reclaim: true}",
                "seeds: [1]",
                "output:",
                f"  directory: {output.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = run_experiment(config)
    json.loads(
        artifacts.runs.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["allocation_policy"]["type"] == "historical-drf"
    for name in (
        "queue-share-timeline.png",
        "borrowed-capacity-timeline.png",
        "fairshare-debt-timeline.png",
        "elastic-replica-timeline.png",
        "fleet-capacity-timeline.png",
    ):
        assert (output / name).stat().st_size > 0


def test_synthetic_multi_seed_experiment_records_distinct_scenarios(tmp_path: Path) -> None:
    output = tmp_path / "multi"
    config = tmp_path / "synthetic.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: synthetic-multi",
                "workload:",
                "  type: synthetic",
                "  generator:",
                "    profile: mixed",
                "    job_count: 8",
                "    node_count: 2",
                "    gpus_per_node: 2",
                "schedulers: [fifo]",
                "seeds: [1, 2]",
                "output:",
                f"  directory: {output.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = run_experiment(config)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))["summary"]

    assert manifest["scenario_hash"] is None
    assert len(manifest["scenario_hashes"]) == 2
    assert {row["runs"] for row in summary} == {2}


def test_tenant_overlay_uses_future_join_capacity_for_limits() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "future",
                    "available": False,
                    "schedulable": False,
                    "gpus": [{"id": "g0", "memory_gb": 40}],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("job", 1, 1, 1, 20)],
        fleet_events=(FleetEvent(1, FleetEventType.NODE_JOIN, "future"),),
    )
    overlaid = _apply_tenant_overlay(scenario, {"tenant_count": 1})
    queue = overlaid.queues[0]
    result = Simulator.from_scenario(overlaid, create_scheduler("drf", overlaid)).run()
    assert queue.guaranteed.gpu_units == 1
    assert queue.limit is not None
    assert queue.limit.gpu_units == 1
    assert queue.limit.gpu_memory_gb == 40
    assert result.jobs[0].first_start_time == 1
    assert result.jobs[0].completion_time == 2


def test_tenant_overlay_does_not_union_disjoint_fleet_capacity() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "current",
                    "gpus": [{"id": "g0", "memory_gb": 40}],
                },
                {
                    "id": "future",
                    "available": False,
                    "schedulable": False,
                    "gpus": [{"id": "g1", "memory_gb": 40}],
                },
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("job", 0, 1, 1, 20)],
        fleet_events=(
            FleetEvent(1, FleetEventType.NODE_FAIL, "current"),
            FleetEvent(2, FleetEventType.NODE_JOIN, "future"),
        ),
    )
    overlaid = _apply_tenant_overlay(scenario, {"tenant_count": 1})
    queue = overlaid.queues[0]
    assert queue.guaranteed.gpu_units == 1
    assert queue.limit is not None
    assert queue.limit.gpu_units == 1
    assert queue.limit.gpu_memory_gb == 40


def test_tenant_overlay_excludes_capacity_removed_at_time_zero() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "removed",
                    "gpus": [{"id": "g0", "memory_gb": 40}],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("job", 0, 1, 1, 20)],
        fleet_events=(FleetEvent(0, FleetEventType.NODE_FAIL, "removed"),),
    )
    overlaid = _apply_tenant_overlay(scenario, {"tenant_count": 1})
    queue = overlaid.queues[0]
    assert queue.guaranteed.gpu_units == 0
    assert queue.limit is not None
    assert queue.limit.gpu_units == 0
    assert queue.limit.gpu_memory_gb == 0
