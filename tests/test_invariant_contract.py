from __future__ import annotations

import time
from pathlib import Path

import pytest

from gpu_scheduler_lab.models import EventType, Job, Priority
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.queues import QueueSpec, ResourceVector
from gpu_scheduler_lab.scenario import Scenario, load_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.simulator.engine import Simulator
from gpu_scheduler_lab.study.invariants import (
    REQUIRED_INVARIANT_IDS,
    InvariantContract,
    generate_baseline,
    render_baseline,
)

ROOT = Path(__file__).resolve().parents[1]


def test_contract_contains_exactly_twelve_required_invariants() -> None:
    contract = InvariantContract.load(ROOT / "study/invariants.yaml")
    assert tuple(item.id for item in contract.invariants) == REQUIRED_INVARIANT_IDS
    assert len(contract.golden_cases) == 3


def test_logical_metric_baseline_is_current() -> None:
    contract = InvariantContract.load(ROOT / "study/invariants.yaml")
    assert render_baseline(generate_baseline(contract)) == contract.baseline_path.read_text(
        encoding="utf-8"
    )


def test_golden_scenarios_have_loose_wall_clock_guardrail() -> None:
    contract = InvariantContract.load(ROOT / "study/invariants.yaml")
    started = time.perf_counter()
    generate_baseline(contract)
    assert time.perf_counter() - started < 30


def test_fairshare_resorts_after_each_allocation() -> None:
    scenario = Scenario(
        Cluster.from_dict(
            {
                "nodes": [
                    {
                        "id": "node",
                        "gpus": [
                            {"id": "g0", "memory_gb": 40},
                            {"id": "g1", "memory_gb": 40},
                        ],
                    }
                ]
            }
        ),
        [
            Job("a-first", 0, 1, 1, 20, queue_id="a"),
            Job("a-second", 0, 1, 1, 20, queue_id="a", priority=Priority.HIGH),
            Job("b-first", 0, 1, 1, 20, queue_id="b"),
        ],
        queues=(
            QueueSpec("a", "root", limit=ResourceVector(2, 80)),
            QueueSpec("b", "root", limit=ResourceVector(2, 80)),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("drf", scenario)).run()
    starts = [record.job_id for record in result.trace if record.event is EventType.JOB_START]
    assert starts[:2] == ["a-second", "b-first"]


def test_old_completion_cannot_release_new_generation() -> None:
    scenario = load_scenario(ROOT / "scenarios/revocable-fleet.yaml")
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    job = result.jobs[0]
    resume = next(record for record in result.trace if record.event is EventType.JOB_RESUME)
    completes = [record for record in result.trace if record.event is EventType.JOB_COMPLETE]
    assert job.run_generation == 3
    assert resume.time == 9
    assert [record.time for record in completes] == [16]
    assert job.completion_time == 16


def test_contract_rejects_duplicate_invariant_ids(tmp_path: Path) -> None:
    source = (ROOT / "study/invariants.yaml").read_text(encoding="utf-8")
    duplicate = source.replace(
        "  - id: atomic-gang-allocation", "  - id: gpu-exclusive-ownership", 1
    )
    config = tmp_path / "study" / "invariants.yaml"
    config.parent.mkdir()
    config.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        InvariantContract.load(config)
