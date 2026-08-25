from __future__ import annotations

from pathlib import Path

import pytest

from gpu_scheduler_lab.elastic import ElasticSpec
from gpu_scheduler_lab.fleet import FleetEvent, FleetEventType
from gpu_scheduler_lab.models import EventType, JobStatus, Priority, TopologyMode
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.queues import QueueSpec, ResourceVector
from gpu_scheduler_lab.scenario import Scenario, load_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.schedulers.fifo import FIFOScheduler
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


@pytest.mark.parametrize("scheduler_name", ["binpack", "spread"])
def test_placement_schedulers_ignore_unavailable_nodes(scheduler_name: str) -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "unavailable",
                    "available": False,
                    "gpus": [{"id": "bad", "memory_gb": 40}],
                },
                {
                    "id": "available",
                    "gpus": [{"id": "good", "memory_gb": 40}],
                },
            ]
        }
    )
    placement = create_scheduler(scheduler_name).place(cluster, Job("job", 0, 1, 1, 20))
    assert placement == ["good"]


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


def test_elastic_reclaim_refreshes_shared_ancestor_usage() -> None:
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
                queue_id="team/research",
                elastic=ElasticSpec(2, 4, 4),
            ),
            Job("product", 1, 2, 2, 20, queue_id="team/product"),
        ],
        queues=(
            QueueSpec("team", "root", limit=ResourceVector(4, 160)),
            QueueSpec(
                "team/research",
                "team",
                guaranteed=ResourceVector(2),
                limit=ResourceVector(4, 160),
            ),
            QueueSpec(
                "team/product",
                "team",
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
    assert borrower.elastic_scale_down_count == 1
    assert borrower.preemption_count == 0


def test_elastic_reclaim_does_not_commit_infeasible_partial_shrinks() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": f"n{node}",
                    "gpus": [{"id": f"g{node}{gpu}", "memory_gb": 40} for gpu in range(2)],
                }
                for node in range(2)
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job(
                f"elastic-{index}",
                0,
                10,
                2,
                20,
                queue_id=f"elastic-{index}",
                topology_mode=TopologyMode.REQUIRE_SAME_NODE,
                elastic=ElasticSpec(1, 2, 2),
            )
            for index in range(2)
        ]
        + [
            Job(
                "target",
                1,
                1,
                2,
                20,
                queue_id="target",
                gang=True,
                topology_mode=TopologyMode.REQUIRE_SAME_NODE,
            )
        ],
        queues=(
            QueueSpec(
                "elastic-0", "root", guaranteed=ResourceVector(1), limit=ResourceVector(2, 80)
            ),
            QueueSpec(
                "elastic-1", "root", guaranteed=ResourceVector(1), limit=ResourceVector(2, 80)
            ),
            QueueSpec("target", "root", guaranteed=ResourceVector(2), limit=ResourceVector(2, 80)),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    elastic_jobs = [job for job in result.jobs if job.id.startswith("elastic-")]
    assert all(job.elastic_scale_down_count == 0 for job in elastic_jobs)


def test_elastic_scale_up_preserves_required_topology() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node-a",
                    "topology": {"zone": "z", "rack": "r-a"},
                    "gpus": [{"id": "a0", "model": "A", "memory_gb": 80}],
                },
                {
                    "id": "node-b",
                    "topology": {"zone": "z", "rack": "r-b"},
                    "gpus": [
                        {"id": "b0", "model": "B", "memory_gb": 40},
                        {"id": "b1", "model": "B", "memory_gb": 40},
                    ],
                },
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job("blocker", 0, 1, 1, 40, gpu_model="B", queue_id="tenant"),
            Job(
                "elastic",
                0,
                4,
                2,
                40,
                queue_id="tenant",
                topology_mode=TopologyMode.REQUIRE_SAME_NODE,
                elastic=ElasticSpec(1, 2, 2),
            ),
        ],
        queues=(
            QueueSpec(
                "tenant",
                "root",
                guaranteed=ResourceVector(3),
                limit=ResourceVector(3, 200),
            ),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    elastic = next(job for job in result.jobs if job.id == "elastic")
    scale_up = next(record for record in result.trace if record.event is EventType.ELASTIC_SCALE_UP)
    assert elastic.elastic_scale_up_count == 1
    assert set(scale_up.node_ids) == {"node-b"}
    assert result.metrics["topology_requirement_violation_count"] == 0


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


def test_draining_running_node_counts_as_utilized_active_capacity() -> None:
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
        [Job("job", 0, 5, 1, 20, queue_id="tenant")],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(1, 40)),),
        fleet_events=(FleetEvent(1, FleetEventType.NODE_DRAIN, "node"),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    assert result.metrics["node_utilization"] == pytest.approx(1.0)


def test_drained_node_leaves_capacity_denominator_after_last_job() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": f"n{index}",
                    "gpus": [{"id": f"g{index}", "memory_gb": 40}],
                }
                for index in range(2)
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job("on-drained-node", 0, 2, 1, 20, queue_id="tenant"),
            Job("on-active-node", 3, 1, 1, 20, queue_id="tenant"),
        ],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(2, 80)),),
        fleet_events=(FleetEvent(1, FleetEventType.NODE_DRAIN, "n0"),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    assert result.metrics["average_gpu_utilization"] == pytest.approx(0.5)
    assert result.metrics["node_utilization"] == pytest.approx(0.5)
    assert result.metrics["idle_gpu_time"] == pytest.approx(3.0)


def test_admission_forgets_consumed_capacity_return_events() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "schedulable": False,
                    "available": False,
                    "gpus": [{"id": "g0", "memory_gb": 40}],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("late", 3, 1, 1, 20, queue_id="tenant")],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(1, 40)),),
        fleet_events=(
            FleetEvent(1, FleetEventType.NODE_RECOVER, "node"),
            FleetEvent(2, FleetEventType.NODE_FAIL, "node"),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    assert result.jobs[0].rejection_reason == "impossible_gpu_request"


def test_event_cleanup_reaches_fixed_point_after_aging_tick_removal() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "a-node",
                    "gpus": [{"id": "a", "model": "A", "memory_gb": 40}],
                },
                {
                    "id": "b-node",
                    "gpus": [{"id": "b", "model": "B", "memory_gb": 40}],
                },
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("job", 0, 100, 1, 20, gpu_model="A")],
        fleet_events=(FleetEvent(1, FleetEventType.NODE_FAIL, "a-node"),),
    )
    fifo = Simulator.from_scenario(scenario, FIFOScheduler()).run()
    preemptive = Simulator.from_scenario(scenario, PreemptiveScheduler()).run()
    for result in (fifo, preemptive):
        assert result.metrics["simulation_horizon"] == pytest.approx(1.0)
        assert result.metrics["average_gpu_utilization"] == pytest.approx(0.5)
        assert result.metrics["idle_gpu_time"] == pytest.approx(1.0)


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


def test_drain_invalidates_and_replans_deferred_reclaim_reservation() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": f"n{index}",
                    "gpus": [{"id": f"g{index}", "memory_gb": 40}],
                }
                for index in range(3)
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job("r0", 0, 10, 1, 20, queue_id="research", checkpoint_cost=2),
            Job("r1", 0, 10, 1, 20, queue_id="research", checkpoint_cost=2),
            Job("target", 1, 1, 2, 20, queue_id="product", gang=True),
        ],
        queues=(
            QueueSpec("research", "root", limit=ResourceVector(3, 120)),
            QueueSpec(
                "product",
                "root",
                guaranteed=ResourceVector(2),
                limit=ResourceVector(2, 80),
            ),
        ),
        fleet_events=(FleetEvent(1.5, FleetEventType.NODE_DRAIN, "n0"),),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    target = next(job for job in result.jobs if job.id == "target")
    assert target.first_start_time == 3.5
    assert target.completion_time == 4.5
    assert all(job.status is JobStatus.COMPLETED for job in result.jobs)


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


def test_elastic_reclaim_releases_only_borrowed_replica_budget() -> None:
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
                "elastic",
                0,
                10,
                4,
                20,
                queue_id="a",
                elastic=ElasticSpec(1, 4, 4),
            ),
            Job("target", 1, 1, 1, 20, queue_id="b"),
        ],
        queues=(
            QueueSpec("a", "root", guaranteed=ResourceVector(3), limit=ResourceVector(4, 160)),
            QueueSpec("b", "root", guaranteed=ResourceVector(1), limit=ResourceVector(1, 40)),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    elastic = next(job for job in result.jobs if job.id == "elastic")
    target = next(job for job in result.jobs if job.id == "target")
    shrink = next(
        record
        for record in result.trace
        if record.event is EventType.ELASTIC_SCALE_DOWN and record.job_id == "elastic"
    )
    assert "replicas=4->3" in shrink.detail
    assert elastic.preemption_count == 0
    assert target.first_start_time == 1


def test_reclaim_combines_elastic_shrink_and_fixed_victim_transactionally() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [{"id": f"g{index}", "memory_gb": 40} for index in range(3)],
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
                10,
                2,
                20,
                priority=Priority.HIGH,
                queue_id="a",
                elastic=ElasticSpec(1, 2, 2),
            ),
            Job("fixed", 0, 10, 1, 20, priority=Priority.LOW, queue_id="a"),
            Job("target", 1, 1, 2, 20, queue_id="b", gang=True),
        ],
        queues=(
            QueueSpec("a", "root", guaranteed=ResourceVector(1), limit=ResourceVector(3, 120)),
            QueueSpec("b", "root", guaranteed=ResourceVector(2), limit=ResourceVector(2, 80)),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    elastic = next(job for job in result.jobs if job.id == "elastic")
    fixed = next(job for job in result.jobs if job.id == "fixed")
    target = next(job for job in result.jobs if job.id == "target")
    assert elastic.elastic_scale_down_count == 1
    assert elastic.preemption_count == 0
    assert fixed.reclaim_victim_count == 1
    assert target.first_start_time == 1


def test_whole_elastic_preemption_supersedes_projected_partial_shrink() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [{"id": f"g{index}", "memory_gb": 40} for index in range(3)],
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
                10,
                3,
                20,
                queue_id="a",
                elastic=ElasticSpec(1, 3, 3),
            ),
            Job("target", 1, 1, 3, 20, queue_id="b", gang=True),
        ],
        queues=(
            QueueSpec("a", "root", limit=ResourceVector(3, 120)),
            QueueSpec("b", "root", guaranteed=ResourceVector(3), limit=ResourceVector(3, 120)),
        ),
    )

    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()

    elastic = next(job for job in result.jobs if job.id == "elastic")
    target = next(job for job in result.jobs if job.id == "target")
    reclaim_shrinks = [
        record
        for record in result.trace
        if record.event is EventType.ELASTIC_SCALE_DOWN
        and record.job_id == "elastic"
        and "PREEMPT_RECLAIM" in record.detail
    ]
    assert reclaim_shrinks == []
    assert elastic.reclaim_victim_count == 1
    assert target.first_start_time == 1


def test_deferred_elastic_reclaim_starts_at_entitled_minimum() -> None:
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
            Job("borrower", 0, 10, 2, 20, queue_id="borrower", checkpoint_cost=2),
            Job("filler", 0, 3, 2, 20, queue_id="filler"),
            Job(
                "target",
                1,
                1,
                4,
                20,
                queue_id="target",
                elastic=ElasticSpec(2, 4, 4),
            ),
        ],
        queues=(
            QueueSpec("borrower", "root", limit=ResourceVector(2, 80)),
            QueueSpec("filler", "root", limit=ResourceVector(2, 80), reclaimable=False),
            QueueSpec("target", "root", guaranteed=ResourceVector(2), limit=ResourceVector(4, 160)),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    target = next(job for job in result.jobs if job.id == "target")
    start = next(
        record
        for record in result.trace
        if record.event is EventType.JOB_START and record.job_id == "target"
    )
    assert target.first_start_time == 3
    assert len(start.gpu_ids) == 2


def test_deferred_reclaim_releases_invalid_quota_claim_and_resumes_victim() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "h-node",
                    "gpus": [{"id": "h", "model": "H100", "memory_gb": 80}],
                },
                {
                    "id": "a-node",
                    "gpus": [{"id": "a", "model": "A10", "memory_gb": 24}],
                },
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job(
                "borrower",
                0,
                20,
                1,
                20,
                gpu_model="H100",
                queue_id="borrower",
                checkpoint_cost=5,
            ),
            Job("filler", 0, 2, 1, 20, gpu_model="A10", queue_id="filler"),
            Job("target", 1, 1, 1, 20, gpu_model="H100", queue_id="entitled"),
            Job("other", 2, 10, 1, 20, gpu_model="A10", queue_id="entitled"),
        ],
        queues=(
            QueueSpec("borrower", "root", limit=ResourceVector(1, 80)),
            QueueSpec("filler", "root", limit=ResourceVector(1, 24)),
            QueueSpec(
                "entitled",
                "root",
                guaranteed=ResourceVector(1),
                limit=ResourceVector(1, 104),
            ),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    resumed_at = [
        record.time
        for record in result.trace
        if record.event is EventType.JOB_RESUME and record.job_id == "borrower"
    ]
    target = next(job for job in result.jobs if job.id == "target")
    assert resumed_at[0] == 6
    assert target.first_start_time == 17


def test_elastic_scale_up_uses_spare_capacity_without_explicit_guarantee() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [{"id": f"g{index}", "memory_gb": 40} for index in range(2)],
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
                2,
                20,
                queue_id="tenant",
                elastic=ElasticSpec(1, 2, 2),
            )
        ],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(2, 80)),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("drf", scenario)).run()
    scale_up = next(record for record in result.trace if record.event is EventType.ELASTIC_SCALE_UP)
    assert scale_up.time == 0
    assert "replicas=1->2" in scale_up.detail


def test_elastic_scale_up_ignores_sibling_without_additional_demand() -> None:
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
            Job("fixed", 0, 10, 1, 20, queue_id="fixed"),
            Job(
                "elastic",
                0,
                10,
                3,
                20,
                queue_id="elastic",
                elastic=ElasticSpec(1, 3, 3),
            ),
        ],
        queues=(
            QueueSpec("fixed", "root", limit=ResourceVector(4, 160)),
            QueueSpec("elastic", "root", limit=ResourceVector(4, 160)),
        ),
    )

    result = Simulator.from_scenario(scenario, create_scheduler("drf", scenario)).run()

    scale_up = next(record for record in result.trace if record.event is EventType.ELASTIC_SCALE_UP)
    assert scale_up.time == 0
    assert scale_up.job_id == "elastic"
    assert "replicas=1->3" in scale_up.detail


def test_elastic_scale_up_uses_hierarchical_weight_order() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "a-node",
                    "gpus": [{"id": "a", "model": "A", "memory_gb": 40}],
                },
                {
                    "id": "b-node",
                    "gpus": [{"id": "b", "model": "B", "memory_gb": 40}],
                },
                {
                    "id": "c-node",
                    "available": False,
                    "schedulable": False,
                    "gpus": [{"id": "c", "model": "C", "memory_gb": 40}],
                },
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job(
                "a-job",
                0,
                10,
                2,
                20,
                allowed_gpu_models=("A", "C"),
                queue_id="a",
                elastic=ElasticSpec(1, 2, 2),
            ),
            Job(
                "z-job",
                0,
                10,
                2,
                20,
                allowed_gpu_models=("B", "C"),
                queue_id="z",
                elastic=ElasticSpec(1, 2, 2),
            ),
        ],
        queues=(
            QueueSpec("a", "root", weight=1, limit=ResourceVector(2, 80)),
            QueueSpec("z", "root", weight=2, limit=ResourceVector(2, 80)),
        ),
        fleet_events=(FleetEvent(1, FleetEventType.NODE_JOIN, "c-node"),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    scale_up = next(record for record in result.trace if record.event is EventType.ELASTIC_SCALE_UP)
    assert scale_up.time == 1
    assert scale_up.job_id == "z-job"


def test_partial_drain_counts_only_grandfathered_gpu_capacity() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "spot",
                    "revocable": True,
                    "gpus": [{"id": f"g{index}", "memory_gb": 40} for index in range(8)],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("job", 0, 5, 1, 20, queue_id="tenant")],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(8, 320)),),
        fleet_events=(FleetEvent(1, FleetEventType.NODE_DRAIN, "spot"),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    assert result.metrics["average_gpu_utilization"] == pytest.approx(5 / 12)
    assert result.metrics["gpu_memory_utilization"] == pytest.approx(100 / 480)
    assert result.metrics["idle_gpu_time"] == pytest.approx(7)
    assert result.metrics["node_utilization"] == pytest.approx(1)
    assert result.metrics["revocable_gpu_time"] == pytest.approx(12)
    at_drain = next(
        point for point in result.metrics["fleet_capacity_timeline"] if point["time"] == 1
    )
    assert at_drain["active_gpus"] == 1
    assert at_drain["revocable_gpus"] == 1


def test_zero_gpu_worker_does_not_dilute_node_utilization() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {"id": "gpu-node", "gpus": [{"id": "g0", "memory_gb": 40}]},
                {"id": "cpu-only", "gpus": []},
            ]
        }
    )
    result = Simulator(cluster, [Job("job", 0, 1, 1, 20)], FIFOScheduler()).run()
    assert result.metrics["average_gpu_utilization"] == pytest.approx(1)
    assert result.metrics["node_utilization"] == pytest.approx(1)


def test_capacity_return_precedes_same_timestamp_admission_and_start() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "spot",
                    "available": False,
                    "schedulable": False,
                    "revocable": True,
                    "gpus": [{"id": "g0", "memory_gb": 40}],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("job", 1, 1, 1, 20, queue_id="tenant")],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(1, 40)),),
        fleet_events=(FleetEvent(1, FleetEventType.CAPACITY_RETURN, "spot"),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    job = result.jobs[0]
    records = [record for record in result.trace if record.time == 1]
    return_index = next(
        index for index, record in enumerate(records) if record.event is EventType.CAPACITY_RETURN
    )
    later = {
        EventType.JOB_ADMIT,
        EventType.JOB_ARRIVAL,
        EventType.JOB_START,
    }
    assert all(
        return_index < index for index, record in enumerate(records) if record.event in later
    )
    assert job.admission_time == 1
    assert job.first_start_time == 1
    assert job.completion_time == 2
