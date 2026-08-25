from __future__ import annotations

import csv
import json
from pathlib import Path

from conftest import make_cluster

from gpu_scheduler_lab.experiments import run_experiment, scenario_hash
from gpu_scheduler_lab.experiments.aggregation import aggregate_runs
from gpu_scheduler_lab.models import Job
from gpu_scheduler_lab.scenario import Scenario, write_scenario


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
    assert len(manifest["runs"]) == 2
    assert len({run["scenario_hash"] for run in manifest["runs"]}) == 1
    with artifacts.summary_csv.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))


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
