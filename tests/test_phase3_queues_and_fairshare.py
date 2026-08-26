from __future__ import annotations

import math
from pathlib import Path

import pytest

from gpu_scheduler_lab.allocation import FairShareScheduler
from gpu_scheduler_lab.elastic import ElasticSpec
from gpu_scheduler_lab.fairshare import AccountingPolicy, DecayedUsageHistory
from gpu_scheduler_lab.fairshare.drf import weighted_dominant_share
from gpu_scheduler_lab.models import EventType, Job, Priority, TopologyMode
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
    with pytest.raises(ValueError, match="child guarantees"):
        QueueHierarchy(
            [
                QueueSpec(
                    "parent",
                    "root",
                    guaranteed=ResourceVector(1),
                    limit=ResourceVector(2, 100),
                ),
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


def test_no_borrow_allows_descendant_to_use_parent_guarantee() -> None:
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
        [Job("child-job", 0, 1, 1, 20, queue_id="team/child")],
        queues=(
            QueueSpec.from_dict(
                {
                    "id": "team",
                    "parent": "root",
                    "guaranteed": {"gpu_units": 1},
                    "limit": {"gpu_units": 1},
                }
            ),
            QueueSpec.from_dict(
                {
                    "id": "team/child",
                    "parent": "team",
                    "limit": {"gpu_units": 1},
                }
            ),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-no-borrow", scenario)
    ).run()
    assert result.jobs[0].completion_time == 1

    nested = QueueHierarchy(
        [
            QueueSpec("parent", "root", guaranteed=ResourceVector(1), limit=ResourceVector(2, 80)),
            QueueSpec(
                "parent/child",
                "parent",
                guaranteed=ResourceVector(1),
                limit=ResourceVector(2, 80),
            ),
        ]
    )
    assert not nested.can_allocate("parent/child", ResourceVector(2, 40), {}, borrowing=False)

    borrowing_disabled = QueueHierarchy(
        [
            QueueSpec(
                "locked",
                "root",
                guaranteed=ResourceVector(1),
                limit=ResourceVector(2, 80),
                borrowing_enabled=False,
            ),
            QueueSpec("locked/child", "locked", limit=ResourceVector(2, 80)),
        ]
    )
    assert not borrowing_disabled.can_allocate(
        "locked/child",
        ResourceVector(1, 20),
        {"locked/child": ResourceVector(1, 20)},
        borrowing=True,
    )


def test_pending_order_prioritizes_descendant_parent_entitlement() -> None:
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
            Job("a-borrower", 0, 10, 1, 20, queue_id="borrower"),
            Job("z-entitled", 0, 1, 1, 20, queue_id="team/child"),
        ],
        queues=(
            QueueSpec("borrower", "root", limit=ResourceVector(1, 40)),
            QueueSpec("team", "root", guaranteed=ResourceVector(1), limit=ResourceVector(1, 40)),
            QueueSpec("team/child", "team", limit=ResourceVector(1, 40)),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("fairshare-borrow", scenario)).run()
    borrower, entitled = result.jobs
    assert entitled.first_start_time == 0
    assert borrower.first_start_time == 1


def test_drf_reranks_simultaneous_jobs_after_each_allocation() -> None:
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
    jobs = [
        Job(f"{queue}-{index}", 0, 10, 1, 20, queue_id=queue)
        for queue in ("a", "b")
        for index in range(4)
    ]
    scenario = Scenario(
        cluster,
        jobs,
        queues=(
            QueueSpec("a", "root", limit=ResourceVector(4, 160)),
            QueueSpec("b", "root", limit=ResourceVector(4, 160)),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("drf", scenario)).run()
    initial = [job for job in result.jobs if job.first_start_time == 0]
    assert sum(job.queue_id == "a" for job in initial) == 2
    assert sum(job.queue_id == "b" for job in initial) == 2


def test_guarantee_first_orders_only_jobs_that_fit_remaining_entitlement() -> None:
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
            Job("a-oversized", 0, 10, 2, 20, queue_id="a", gang=True),
            Job("z-entitled", 0, 1, 1, 20, queue_id="b"),
        ],
        queues=(
            QueueSpec("a", "root", guaranteed=ResourceVector(1), limit=ResourceVector(2, 80)),
            QueueSpec("b", "root", guaranteed=ResourceVector(1), limit=ResourceVector(1, 40)),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("fairshare-borrow", scenario)).run()
    oversized, entitled = result.jobs
    assert entitled.first_start_time == 0
    assert oversized.first_start_time == 1


def test_hierarchical_drf_applies_parent_weight_to_descendants() -> None:
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
    jobs = [
        Job(f"{branch}-{index}", 0, 10, 1, 20, queue_id=f"{branch}/leaf")
        for branch in ("a", "b")
        for index in range(3)
    ]
    scenario = Scenario(
        cluster,
        jobs,
        queues=(
            QueueSpec("a", "root", weight=2, limit=ResourceVector(3, 120)),
            QueueSpec("a/leaf", "a", limit=ResourceVector(3, 120)),
            QueueSpec("b", "root", weight=1, limit=ResourceVector(3, 120)),
            QueueSpec("b/leaf", "b", limit=ResourceVector(3, 120)),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("drf", scenario)).run()
    initial = [job for job in result.jobs if job.first_start_time == 0]
    assert sum(job.queue_id == "a/leaf" for job in initial) == 2
    assert sum(job.queue_id == "b/leaf" for job in initial) == 1


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


def test_quota_admission_sums_cheapest_feasible_gpu_weights() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [
                        {"id": "light", "model": "L", "memory_gb": 40},
                        {"id": "heavy", "model": "H", "memory_gb": 40},
                    ],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [Job("gang", 0, 1, 2, 20, queue_id="tenant", gang=True)],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(2, 80)),),
        accounting=AccountingPolicy({"L": 1, "H": 2}),
        admission_mode="quota-aware",
    )
    result = Simulator.from_scenario(scenario, create_scheduler("fairshare-borrow", scenario)).run()
    assert result.jobs[0].rejection_reason == "queue_hard_limit"


@pytest.mark.parametrize(
    "topology_mode",
    [TopologyMode.REQUIRE_SAME_NODE, TopologyMode.REQUIRE_SAME_RACK],
)
def test_admission_rejects_topologically_impossible_minimum(
    topology_mode: TopologyMode,
) -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": f"n{index}",
                    "topology": {"zone": "z", "rack": f"r{index}"},
                    "gpus": [{"id": f"g{index}", "memory_gb": 40}],
                }
                for index in range(2)
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job(
                "gang",
                0,
                1,
                2,
                20,
                queue_id="tenant",
                gang=True,
                topology_mode=topology_mode,
            )
        ],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(2, 80)),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("fairshare-borrow", scenario)).run()
    assert result.jobs[0].rejection_reason == "impossible_gpu_request"


def test_quota_admission_uses_cheapest_topology_feasible_gpu_set() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "cheap-a",
                    "gpus": [{"id": "l0", "model": "L", "memory_gb": 40}],
                },
                {
                    "id": "cheap-b",
                    "gpus": [{"id": "l1", "model": "L", "memory_gb": 40}],
                },
                {
                    "id": "heavy",
                    "gpus": [
                        {"id": "h0", "model": "H", "memory_gb": 40},
                        {"id": "h1", "model": "H", "memory_gb": 40},
                    ],
                },
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job(
                "gang",
                0,
                1,
                2,
                20,
                queue_id="tenant",
                gang=True,
                topology_mode=TopologyMode.REQUIRE_SAME_NODE,
            )
        ],
        queues=(QueueSpec("tenant", "root", limit=ResourceVector(2, 80)),),
        accounting=AccountingPolicy({"L": 1, "H": 2}),
        admission_mode="quota-aware",
    )
    result = Simulator.from_scenario(scenario, create_scheduler("fairshare-borrow", scenario)).run()
    assert result.jobs[0].rejection_reason == "queue_hard_limit"


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


def test_reclaim_restores_parent_entitlement_for_descendant() -> None:
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
            Job("borrower", 0, 10, 2, 20, queue_id="research"),
            Job("child", 1, 1, 1, 20, queue_id="team/child"),
        ],
        queues=(
            QueueSpec("research", "root", limit=ResourceVector(2, 80)),
            QueueSpec("team", "root", guaranteed=ResourceVector(1), limit=ResourceVector(1, 40)),
            QueueSpec("team/child", "team", limit=ResourceVector(1, 40)),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    borrower, child = result.jobs
    assert child.first_start_time == 1
    assert borrower.reclaim_victim_count == 1


def test_reclaim_uses_parent_entitlement_when_leaf_guarantee_is_smaller() -> None:
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
            Job("outside", 0, 10, 2, 20, queue_id="outside"),
            Job("prod-gang", 1, 1, 2, 20, queue_id="team/prod", gang=True),
        ],
        queues=(
            QueueSpec(
                "team",
                "root",
                guaranteed=ResourceVector(2),
                limit=ResourceVector(2, 80),
            ),
            QueueSpec(
                "team/prod",
                "team",
                guaranteed=ResourceVector(1),
                limit=ResourceVector(2, 80),
            ),
            QueueSpec("outside", "root", limit=ResourceVector(2, 80)),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    outside, prod = result.jobs
    assert outside.reclaim_victim_count == 1
    assert prod.first_start_time == 1


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


def test_reclaim_target_must_fit_remaining_entitlement() -> None:
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
            Job("borrower", 0, 10, 2, 20, queue_id="borrower"),
            Job("oversized", 1, 1, 2, 20, queue_id="entitled", gang=True),
        ],
        queues=(
            QueueSpec("borrower", "root", limit=ResourceVector(2, 80)),
            QueueSpec(
                "entitled",
                "root",
                guaranteed=ResourceVector(1),
                limit=ResourceVector(2, 80),
            ),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    borrower, oversized = result.jobs
    assert borrower.preemption_count == 0
    assert oversized.first_start_time == 10


def test_reclaim_uses_contended_sibling_branch_entitlement() -> None:
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
            Job("borrower", 0, 10, 1, 20, queue_id="team/borrow"),
            Job("entitled", 1, 1, 1, 20, queue_id="team/entitled"),
        ],
        queues=(
            QueueSpec("team", "root", guaranteed=ResourceVector(1), limit=ResourceVector(1, 40)),
            QueueSpec("team/borrow", "team", limit=ResourceVector(1, 40)),
            QueueSpec(
                "team/entitled",
                "team",
                guaranteed=ResourceVector(1),
                limit=ResourceVector(1, 40),
            ),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    borrower, entitled = result.jobs
    assert borrower.reclaim_victim_count == 1
    assert entitled.first_start_time == 1


def test_reclaim_uses_aggregate_branch_excess_not_leaf_borrow_marker() -> None:
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
            Job("team-job", 0, 10, 2, 20, queue_id="team/leaf"),
            Job("product", 1, 1, 1, 20, queue_id="product"),
        ],
        queues=(
            QueueSpec("team", "root", limit=ResourceVector(2, 80)),
            QueueSpec(
                "team/leaf",
                "team",
                guaranteed=ResourceVector(2),
                limit=ResourceVector(2, 80),
            ),
            QueueSpec(
                "product",
                "root",
                guaranteed=ResourceVector(1),
                limit=ResourceVector(1, 40),
            ),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    team_job, product = result.jobs
    assert team_job.borrowed_gpu_units == 0
    assert team_job.reclaim_victim_count == 1
    assert product.first_start_time == 1


def test_reclaim_does_not_exceed_borrowed_branch_budget() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [
                        {"id": f"h{index}", "model": "H100", "memory_gb": 80} for index in range(3)
                    ]
                    + [{"id": "a0", "model": "A10", "memory_gb": 24}],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            *[
                Job(
                    f"a-{index}",
                    0,
                    10,
                    1,
                    20,
                    gpu_model="H100",
                    queue_id="a",
                )
                for index in range(3)
            ],
            Job(
                "target",
                1,
                1,
                2,
                20,
                gpu_model="H100",
                queue_id="b",
                gang=True,
            ),
        ],
        queues=(
            QueueSpec("a", "root", guaranteed=ResourceVector(2), limit=ResourceVector(3, 240)),
            QueueSpec("b", "root", guaranteed=ResourceVector(2), limit=ResourceVector(2, 160)),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    a_jobs = [job for job in result.jobs if job.queue_id == "a"]
    target = next(job for job in result.jobs if job.id == "target")
    assert all(job.reclaim_victim_count == 0 for job in a_jobs)
    assert target.first_start_time == 10


def test_historical_fairshare_penalizes_prior_service() -> None:
    instantaneous = _run("scenarios/historical-fairshare.yaml", "drf")
    historical = _run("scenarios/historical-fairshare.yaml", "historical-drf")
    instant_b = next(job for job in instantaneous.jobs if job.id == "b-late")
    historical_b = next(job for job in historical.jobs if job.id == "b-late")
    assert instant_b.first_start_time == 15
    assert historical_b.first_start_time == 10
    assert historical.metrics["queue_metrics"]["tenant-a"]["fairshare_debt"] > 0


def test_historical_service_records_complete_interval_without_middle_event() -> None:
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
        [Job("job", 0, 10, 1, 20, queue_id="a")],
        queues=(QueueSpec("a", "root", limit=ResourceVector(1, 40)),),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    expected = 300 * (1 - 0.5 ** (10 / 300)) / math.log(2)
    assert result.metrics["queue_metrics"]["a"]["historical_service"] == pytest.approx(expected)


def test_historical_service_is_invariant_to_interval_splitting() -> None:
    single = DecayedUsageHistory(half_life=10)
    single.integrate(10, {"a": 1})

    split = DecayedUsageHistory(half_life=10)
    for now in range(1, 11):
        split.integrate(now, {"a": 1})

    assert split.service["a"] == pytest.approx(single.service["a"])


def test_historical_parent_debt_orders_descendant_jobs() -> None:
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
            Job("a-first", 0, 10, 1, 20, queue_id="research/leaf"),
            Job("a-second", 0, 1, 1, 20, queue_id="research/leaf"),
            Job("b-late", 5, 1, 1, 20, queue_id="product/leaf"),
        ],
        queues=(
            QueueSpec("research", "root", limit=ResourceVector(1, 40)),
            QueueSpec("research/leaf", "research", limit=ResourceVector(1, 40)),
            QueueSpec("product", "root", limit=ResourceVector(1, 40)),
            QueueSpec("product/leaf", "product", limit=ResourceVector(1, 40)),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    a_second = next(job for job in result.jobs if job.id == "a-second")
    b_late = next(job for job in result.jobs if job.id == "b-late")
    assert b_late.first_start_time == 10
    assert a_second.first_start_time == 11


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


def test_deferred_reclaim_revalidates_entitlement_before_reserved_start() -> None:
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
                limit=ResourceVector(2, 104),
            ),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    borrower = next(job for job in result.jobs if job.id == "borrower")
    target = next(job for job in result.jobs if job.id == "target")
    other = next(job for job in result.jobs if job.id == "other")
    assert other.first_start_time == 2
    assert target.first_start_time is not None and target.first_start_time >= 12
    assert borrower.reclaim_victim_count >= 1


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


def test_admission_rejects_unavailable_capacity_without_future_return() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "failed",
                    "available": False,
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


def test_fairshare_tie_breaks_by_job_priority_before_queue_id() -> None:
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
            Job("a-low", 0, 1, 1, 20, priority=Priority.LOW, queue_id="a"),
            Job("z-critical", 0, 1, 1, 20, priority=Priority.CRITICAL, queue_id="z"),
        ],
        queues=(
            QueueSpec("a", "root", limit=ResourceVector(1, 40)),
            QueueSpec("z", "root", limit=ResourceVector(1, 40)),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("drf", scenario)).run()
    low, critical = result.jobs
    assert critical.first_start_time == 0
    assert low.first_start_time == 1


def test_guarantee_pass_validates_concrete_weighted_placement() -> None:
    cluster = Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [
                        {"id": "cheap", "model": "cheap", "memory_gb": 40},
                        {"id": "expensive", "model": "expensive", "memory_gb": 40},
                    ],
                }
            ]
        }
    )
    scenario = Scenario(
        cluster,
        [
            Job("filler", 0, 10, 1, 20, gpu_model="cheap", queue_id="filler"),
            Job(
                "a-flex",
                1,
                1,
                1,
                20,
                allowed_gpu_models=("cheap", "expensive"),
                queue_id="a",
            ),
            Job("b-exact", 1, 1, 1, 20, gpu_model="expensive", queue_id="b"),
        ],
        queues=(
            QueueSpec("filler", "root", limit=ResourceVector(2, 80)),
            QueueSpec("a", "root", guaranteed=ResourceVector(1), limit=ResourceVector(2, 80)),
            QueueSpec("b", "root", guaranteed=ResourceVector(2), limit=ResourceVector(2, 80)),
        ),
        accounting=AccountingPolicy({"cheap": 1, "expensive": 2}),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("fairshare-borrow", scenario)).run()
    flex = next(job for job in result.jobs if job.id == "a-flex")
    exact = next(job for job in result.jobs if job.id == "b-exact")
    assert exact.first_start_time == 1
    assert flex.first_start_time == 2


def test_elastic_job_starts_at_min_before_sibling_guarantee_is_borrowed() -> None:
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
                "a-elastic",
                0,
                10,
                2,
                20,
                queue_id="a",
                elastic=ElasticSpec(1, 2, 2),
            ),
            Job("b-fixed", 0, 1, 1, 20, queue_id="b"),
        ],
        queues=(
            QueueSpec("a", "root", guaranteed=ResourceVector(1), limit=ResourceVector(2, 80)),
            QueueSpec("b", "root", guaranteed=ResourceVector(1), limit=ResourceVector(1, 40)),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("fairshare-borrow", scenario)).run()
    elastic_start = next(
        record
        for record in result.trace
        if record.event is EventType.JOB_START and record.job_id == "a-elastic"
    )
    fixed = next(job for job in result.jobs if job.id == "b-fixed")
    assert len(elastic_start.gpu_ids) == 1
    assert fixed.first_start_time == 0


def test_hierarchical_reclaim_transfers_only_one_sibling_allocation() -> None:
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
            *[Job(f"a-{index}", 0, 10, 1, 20, queue_id="team/a") for index in range(3)],
            Job("target", 1, 1, 1, 20, queue_id="team/b"),
        ],
        queues=(
            QueueSpec("team", "root", guaranteed=ResourceVector(2), limit=ResourceVector(3, 120)),
            QueueSpec("team/a", "team", guaranteed=ResourceVector(1), limit=ResourceVector(3, 120)),
            QueueSpec("team/b", "team", guaranteed=ResourceVector(1), limit=ResourceVector(1, 40)),
        ),
    )
    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    victims = [job for job in result.jobs if job.queue_id == "team/a"]
    target = next(job for job in result.jobs if job.id == "target")
    assert sum(job.reclaim_victim_count for job in victims) == 1
    assert target.first_start_time == 1


def test_guarantee_satisfaction_ignores_intervals_without_runnable_demand() -> None:
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
            Job("a-short", 0, 1, 1, 20, queue_id="a"),
            Job("b-long", 0, 10, 1, 20, queue_id="b"),
        ],
        queues=(
            QueueSpec("a", "root", guaranteed=ResourceVector(1), limit=ResourceVector(1, 40)),
            QueueSpec("b", "root", guaranteed=ResourceVector(1), limit=ResourceVector(1, 40)),
        ),
    )
    result = Simulator.from_scenario(scenario, create_scheduler("drf", scenario)).run()
    assert result.metrics["queue_metrics"]["a"]["guaranteed_share_satisfaction"] == pytest.approx(1)
    assert result.metrics["queue_metrics"]["b"]["guaranteed_share_satisfaction"] == pytest.approx(1)


def test_guarantee_satisfaction_caps_elastic_demand_at_potential_capacity() -> None:
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
                10,
                4,
                20,
                queue_id="elastic",
                elastic=ElasticSpec(1, 4, 4),
            ),
            Job("borrower", 0, 40, 1, 20, queue_id="borrower"),
        ],
        queues=(
            QueueSpec("elastic", "root", guaranteed=ResourceVector(2)),
            QueueSpec("borrower", "root", limit=ResourceVector(2, 80)),
        ),
    )

    result = Simulator.from_scenario(scenario, create_scheduler("drf", scenario)).run()

    satisfaction = result.metrics["queue_metrics"]["elastic"]["guaranteed_share_satisfaction"]
    assert satisfaction == pytest.approx(0.5)
