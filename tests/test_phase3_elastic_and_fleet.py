from __future__ import annotations

from pathlib import Path

import pytest

from gpu_scheduler_lab.elastic import ElasticSpec
from gpu_scheduler_lab.fleet import FleetEvent, FleetEventType
from gpu_scheduler_lab.models import EventType, JobStatus, Priority
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.queues import QueueSpec, ResourceVector
from gpu_scheduler_lab.scenario import Scenario, load_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.schedulers.preemptive import PreemptiveScheduler
from gpu_scheduler_lab.simulator.engine import SimulationResult, Simulator

ROOT = Path(__file__).resolve().parents[1]


def _run(name: str) -> SimulationResult:
    scenario = load_scenario(ROOT / "scenarios" / name)
    return Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()


def test_elastic_spec_rejects_invalid_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="min"):
        ElasticSpec(4, 2, 8)
    with pytest.raises(ValueError, match="finite"):
        ElasticSpec(1, 1, 2, {2: float("nan")})


def test_elastic_gang_scales_atomically_and_conserves_work() -> None:
    elastic = _run("elastic-gang.yaml")
    fixed = _run("fixed-gang-changing-capacity.yaml")
    job = elastic.jobs[0]
    fixed_job = fixed.jobs[0]
    assert job.status is JobStatus.COMPLETED
    assert job.first_start_time == 0
    assert job.completion_time == pytest.approx(12.5)
    assert fixed_job.first_start_time == 5
    assert fixed_job.completion_time == 15
    assert job.productive_work_completed == pytest.approx(job.total_productive_work)
    assert job.elastic_scale_up_count == 1
    assert elastic.metrics["time_below_preferred"] == pytest.approx(5)
    assert elastic.metrics["simulation_horizon"] == pytest.approx(job.completion_time)
    assert elastic.metrics["average_gpu_utilization"] == pytest.approx(1.0)
    assert any(record.event is EventType.ELASTIC_SCALE_UP for record in elastic.trace)


def test_revocation_recovers_with_preserved_optimistic_work() -> None:
    revocable = _run("revocable-fleet.yaml")
    stable = _run("stable-fleet.yaml")
    job = revocable.jobs[0]
    assert stable.jobs[0].completion_time == 10
    assert job.completion_time == 16
    assert job.accumulated_runtime == pytest.approx(job.duration)
    assert job.recovery_count == 1
    assert revocable.metrics["capacity_revoke_count"] == 1
    assert revocable.metrics["capacity_return_count"] == 1
    assert revocable.metrics["jobs_affected_by_capacity_loss"] == 1
    assert revocable.metrics["recovery_overhead"] == 1


def test_completion_precedes_same_timestamp_revocation() -> None:
    scenario = load_scenario(ROOT / "scenarios/revocable-fleet.yaml")
    scenario.fleet_events = tuple(
        type(event)(10, event.event_type, event.node_id) for event in scenario.fleet_events[:1]
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    assert result.jobs[0].completion_time == 10
    assert result.jobs[0].recovery_count == 0


def test_failed_or_revoked_nodes_never_receive_new_placement() -> None:
    result = _run("revocable-fleet.yaml")
    revoke_index = next(
        index
        for index, record in enumerate(result.trace)
        if record.event is EventType.CAPACITY_REVOKE
    )
    before_return = result.trace[revoke_index + 1 :]
    for record in before_return:
        if record.event is EventType.CAPACITY_RETURN:
            break
        if record.event in {EventType.JOB_START, EventType.JOB_RESUME}:
            assert not {"p0", "p1"}.intersection(record.gpu_ids)


def test_elastic_borrowed_capacity_scales_down_before_job_preemption() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [{"id": f"g{index}", "memory_gb": 40} for index in range(4)],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job(
                "elastic-borrower",
                0,
                10,
                4,
                20,
                queue_id="research",
                elastic=ElasticSpec(2, 4, 4),
            ),
            Job("product", 1, 2, 2, 20, queue_id="product"),
        ],
        queues=(
            QueueSpec(
                "research",
                "root",
                guaranteed=ResourceVector(2),
                limit=ResourceVector(4, 160),
            ),
            QueueSpec(
                "product",
                "root",
                guaranteed=ResourceVector(2),
                limit=ResourceVector(2, 80),
            ),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    borrower, product = result.jobs
    assert product.first_start_time == 1
    assert borrower.preemption_count == 0
    assert borrower.elastic_scale_down_count == 1
    assert any(
        record.event is EventType.ELASTIC_SCALE_DOWN and "PREEMPT_RECLAIM" in record.detail
        for record in result.trace
    )


def test_join_drain_fail_and_recover_have_explicit_semantics() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "dynamic",
                    "schedulable": False,
                    "available": False,
                    "gpus": [
                        {"id": "g0", "memory_gb": 40},
                        {"id": "g1", "memory_gb": 40},
                    ],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("job", 0, 5, 2, 20, queue_id="tenant", restart_cost=1)],
        queues=(
            QueueSpec(
                "tenant",
                "root",
                guaranteed=ResourceVector(2),
                limit=ResourceVector(2, 80),
            ),
        ),
        fleet_events=(
            FleetEvent(0, FleetEventType.NODE_JOIN, "dynamic"),
            FleetEvent(1, FleetEventType.NODE_DRAIN, "dynamic"),
            FleetEvent(2, FleetEventType.NODE_FAIL, "dynamic"),
            FleetEvent(4, FleetEventType.NODE_RECOVER, "dynamic"),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    job = result.jobs[0]
    assert job.first_start_time == 0
    assert job.completion_time == 8
    assert job.recovery_count == 1
    assert result.metrics["node_join_count"] == 1
    assert result.metrics["node_drain_count"] == 1
    assert result.metrics["node_failure_count"] == 1


def test_new_phase3_numeric_fields_reject_nan_and_inf() -> None:
    with pytest.raises(ValueError, match="finite"):
        QueueSpec("tenant", "root", weight=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        ResourceVector(float("nan"), 0)
    with pytest.raises(ValueError, match="finite"):
        FleetEvent(float("nan"), FleetEventType.NODE_FAIL, "node")


def test_elastic_scale_up_never_exceeds_queue_limit() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [{"id": f"g{index}", "memory_gb": 40} for index in range(8)],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job(
                "elastic",
                0,
                4,
                8,
                20,
                queue_id="tenant",
                elastic=ElasticSpec(2, 8, 8),
            )
        ],
        queues=(
            QueueSpec(
                "tenant",
                "root",
                guaranteed=ResourceVector(3),
                limit=ResourceVector(4, 160),
            ),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    assert result.metrics["queue_metrics"]["tenant"]["peak_gpu_usage"] <= 4


def test_node_failure_during_checkpoint_keeps_reclaim_reservation_safe() -> None:
    cluster = Cluster.from_dict(
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
    )
    scenario = Scenario(
        cluster,
        [
            Job(
                "borrower",
                0,
                10,
                2,
                20,
                queue_id="research",
                checkpoint_cost=2,
            ),
            Job("target", 1, 1, 2, 20, queue_id="product"),
        ],
        queues=(
            QueueSpec("research", "root", limit=ResourceVector(2, 80)),
            QueueSpec(
                "product",
                "root",
                guaranteed=ResourceVector(2),
                limit=ResourceVector(2, 80),
            ),
        ),
        fleet_events=(
            FleetEvent(1.5, FleetEventType.NODE_FAIL, "node"),
            FleetEvent(4, FleetEventType.NODE_RECOVER, "node"),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    borrower, target = result.jobs
    assert target.first_start_time == 4
    assert borrower.status is JobStatus.COMPLETED
    assert target.status is JobStatus.COMPLETED


def test_node_failure_during_restart_invalidates_old_restart_event() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [{"id": "g0", "memory_gb": 40}],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job("low", 0, 10, 1, 20, priority=Priority.LOW, restart_cost=3),
            Job("high", 1, 1, 1, 20, priority=Priority.HIGH),
        ],
        fleet_events=(
            FleetEvent(3, FleetEventType.NODE_FAIL, "node"),
            FleetEvent(4, FleetEventType.NODE_RECOVER, "node"),
        ),
    )
    result = Simulator.from_scenario(scenario, PreemptiveScheduler()).run()
    low = next(job for job in result.jobs if job.id == "low")
    assert low.status is JobStatus.COMPLETED
    assert low.recovery_count == 1
    assert low.completion_time == 16
