from __future__ import annotations

import math
from pathlib import Path

import pytest

from gpu_scheduler_lab.allocation import FairShareScheduler
from gpu_scheduler_lab.fairshare import AccountingPolicy
from gpu_scheduler_lab.fairshare.drf import weighted_dominant_share
from gpu_scheduler_lab.models import EventType, Job
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.queues import QueueHierarchy, QueueSpec, ResourceVector
from gpu_scheduler_lab.scenario import Scenario, load_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.simulator.engine import SimulationResult, Simulator

ROOT = Path(__file__).resolve().parents[1]


def _run(path: str, scheduler: str) -> SimulationResult:
    scenario = load_scenario(ROOT / path)
    return Simulator.from_scenario(scenario, create_scheduler(scheduler, scenario)).run()


def test_queue_hierarchy_rejects_invalid_graphs_and_quotas() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        QueueHierarchy([QueueSpec("a", "root"), QueueSpec("a", "root")])
    with pytest.raises(ValueError, match="missing parent"):
        QueueHierarchy([QueueSpec("a", "missing")])
    with pytest.raises(ValueError, match="cycle"):
        QueueHierarchy([QueueSpec("a", "b"), QueueSpec("b", "a")])
    with pytest.raises(ValueError, match="guarantee exceeds"):
        QueueSpec("a", "root", guaranteed=ResourceVector(2), limit=ResourceVector(1))
    with pytest.raises(ValueError, match="child guarantees"):
        QueueHierarchy(
            [
                QueueSpec("parent", "root", limit=ResourceVector(1, 100)),
                QueueSpec("child", "parent", guaranteed=ResourceVector(2)),
            ]
        )


def test_deep_hierarchy_enforces_ancestor_limit() -> None:
    hierarchy = QueueHierarchy(
        [
            QueueSpec("team", "root", limit=ResourceVector(2, 100)),
            QueueSpec("team/research", "team", limit=ResourceVector(4, 100)),
        ]
    )
    assert hierarchy.can_allocate("team/research", ResourceVector(2, 20), {}, borrowing=True)
    assert not hierarchy.can_allocate("team/research", ResourceVector(3, 20), {}, borrowing=True)


def test_no_borrow_honors_only_configured_guarantee_dimensions() -> None:
    queue = QueueSpec.from_dict(
        {
            "id": "tenant",
            "parent": "root",
            "guaranteed": {"gpu_units": 1},
            "limit": {"gpu_units": 1},
        }
    )
    scenario = Scenario(
        Cluster.from_dict(
            {
                "nodes": [
                    {
                        "id": "node",
                        "gpus": [{"id": "g0", "memory_gb": 40}],
                    }
                ]
            }
        ),
        [Job("job", 0, 1, 1, 20, queue_id="tenant")],
        queues=(queue,),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-no-borrow", scenario)
    ).run()
    assert result.jobs[0].first_start_time == 0
    assert result.jobs[0].completion_time == 1
    assert queue.to_dict()["guaranteed"] == {"gpu_units": 1.0}

    zero_gpu = QueueSpec.from_dict({"id": "zero", "parent": "root", "guaranteed": {"gpu_units": 0}})
    assert not QueueHierarchy([zero_gpu]).can_allocate(
        "zero", ResourceVector(1, 20), {}, borrowing=False
    )


def test_weighted_drf_is_finite_and_weight_adjusted() -> None:
    capacity = ResourceVector(8, 320)
    usage = ResourceVector(4, 80)
    assert weighted_dominant_share(usage, capacity, 2) == pytest.approx(0.25)
    assert math.isfinite(weighted_dominant_share(usage, capacity, 2))


def test_selected_gpu_model_cannot_exceed_queue_limit() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "light-node",
                    "gpus": [{"id": "light", "model": "L", "memory_gb": 80}],
                },
                {
                    "id": "heavy-node",
                    "gpus": [{"id": "heavy", "model": "H", "memory_gb": 40}],
                },
            ]
        }
    )
    job = Job("weighted", 0, 1, 1, 40, queue_id="tenant")
    scheduler = FairShareScheduler(
        QueueHierarchy([QueueSpec("tenant", "root", limit=ResourceVector(1, 80))]),
        AccountingPolicy({"L": 1, "H": 2}),
    )
    scheduler.prepare(0, cluster, [job], [])
    assert scheduler.placement.place(cluster, job) == ["heavy"]
    assert scheduler.place(cluster, job) is None


def test_borrowing_and_reclaim_restore_guarantee() -> None:
    borrowing = _run("scenarios/multi-tenant-borrow-reclaim.yaml", "fairshare-borrow")
    reclaim = _run("scenarios/multi-tenant-borrow-reclaim.yaml", "fairshare-reclaim")
    borrower = next(job for job in reclaim.jobs if job.id == "research-borrower")
    product = next(job for job in reclaim.jobs if job.id == "product-guarantee")
    assert borrower.borrowed_gpu_units == 2
    assert borrower.reclaim_victim_count == 1
    assert product.first_start_time == 3
    borrowing_product = next(job for job in borrowing.jobs if job.id == "product-guarantee")
    assert product.first_start_time is not None
    assert borrowing_product.first_start_time is not None
    assert product.first_start_time < borrowing_product.first_start_time
    assert any(
        record.event is EventType.JOB_PREEMPT and "PREEMPT_RECLAIM" in record.detail
        for record in reclaim.trace
    )
    assert reclaim.metrics["queue_metrics"]["research"]["borrowed_gpu_time"] > 0


def test_reclaim_does_not_take_in_guarantee_work_for_peer_borrowing() -> None:
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
            Job("a-in-quota", 0, 10, 1, 20, queue_id="a"),
            Job("b-wants-borrow", 1, 1, 2, 20, queue_id="b", gang=True),
        ],
        queues=(
            QueueSpec("a", "root", guaranteed=ResourceVector(1), limit=ResourceVector(2, 100)),
            QueueSpec("b", "root", guaranteed=ResourceVector(1), limit=ResourceVector(2, 100)),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    a_job, b_job = result.jobs
    assert a_job.reclaim_victim_count == 0
    assert b_job.first_start_time == 10


def test_historical_fairshare_penalizes_prior_service() -> None:
    instantaneous = _run("scenarios/historical-fairshare.yaml", "drf")
    historical = _run("scenarios/historical-fairshare.yaml", "historical-drf")
    instant_b = next(job for job in instantaneous.jobs if job.id == "b-late")
    historical_b = next(job for job in historical.jobs if job.id == "b-late")
    assert instant_b.first_start_time == 15
    assert historical_b.first_start_time == 10
    assert historical.metrics["queue_metrics"]["tenant-a"]["fairshare_debt"] > 0


def test_quota_aware_admission_rejects_impossible_limit_not_busy_capacity() -> None:
    scenario = load_scenario(ROOT / "scenarios/multi-tenant-borrow-reclaim.yaml")
    oversized = Job("oversized", 0, 1, 5, 40, queue_id="research")
    scenario.jobs.append(oversized)
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    rejected = next(job for job in result.jobs if job.id == "oversized")
    product = next(job for job in result.jobs if job.id == "product-guarantee")
    assert rejected.rejection_reason == "impossible_gpu_request"
    assert product.admission_time == product.arrival_time
    assert result.metrics["submitted_job_count"] == 3
    assert result.metrics["rejected_job_count"] == 1


def test_multi_victim_reclaim_reserves_complete_gang_placement() -> None:
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
            Job("r-a", 0, 20, 2, 20, queue_id="research", checkpoint_cost=1),
            Job("r-b", 0, 20, 2, 20, queue_id="research", checkpoint_cost=2),
            Job("product-gang", 1, 2, 4, 20, queue_id="product", gang=True),
            Job("product-peer", 2, 1, 1, 20, queue_id="product"),
        ],
        queues=(
            QueueSpec("research", "root", limit=ResourceVector(4, 160)),
            QueueSpec(
                "product",
                "root",
                guaranteed=ResourceVector(4),
                limit=ResourceVector(4, 160),
            ),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    target = next(job for job in result.jobs if job.id == "product-gang")
    peer = next(job for job in result.jobs if job.id == "product-peer")
    victims = [job for job in result.jobs if job.id.startswith("r-")]
    assert target.first_start_time == 3
    assert target.completion_time is not None
    assert peer.first_start_time is not None and peer.first_start_time >= target.completion_time
    assert [job.reclaim_victim_count for job in victims] == [1, 1]


def test_zero_job_multi_tenant_scenario_serializes_strict_json() -> None:
    scenario = Scenario(
        Cluster.from_dict({"nodes": []}),
        [],
        queues=(QueueSpec("tenant", "root", guaranteed=ResourceVector()),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    assert result.metrics["completion_rate"] == 1
    assert result.metrics["submitted_job_count"] == 0
    import json

    json.dumps(result.to_dict(), allow_nan=False)


def test_admission_rejects_permanently_cordoned_only_capacity() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "cordoned",
                    "schedulable": False,
                    "gpus": [{"id": "g0", "memory_gb": 40}],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("job", 0, 1, 1, 20, queue_id="tenant")],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(1, 40)),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    assert result.jobs[0].rejection_reason == "impossible_gpu_request"
