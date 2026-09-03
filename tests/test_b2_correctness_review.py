from __future__ import annotations

from typing import Any

import pytest

from gpu_scheduler_lab.admission import AdmissionController
from gpu_scheduler_lab.cli import DEFAULT_COMPARE_SCHEDULERS, SCHEDULERS, build_parser
from gpu_scheduler_lab.elastic import ElasticSpec
from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.heterogeneous.study import _with_vendor_outage
from gpu_scheduler_lab.models import (
    GPU,
    AcceleratorKind,
    AcceleratorVendor,
    Cluster,
    Job,
    Node,
    Priority,
    TopologyMode,
)
from gpu_scheduler_lab.queues import QueueHierarchy, QueueSpec, ResourceVector
from gpu_scheduler_lab.scenario import Scenario
from gpu_scheduler_lab.schedulers import PreemptiveScheduler, create_scheduler
from gpu_scheduler_lab.simulator.engine import Simulator, _plan_reclaim_action_key


def _gpu(
    device_id: str,
    node_id: str,
    vendor: AcceleratorVendor,
    kind: AcceleratorKind,
) -> GPU:
    return GPU(
        id=device_id,
        node_id=node_id,
        memory_capacity_gb=64,
        model="A100-80GB" if vendor is AcceleratorVendor.NVIDIA else "Ascend-910B",
        vendor=vendor,
        kind=kind,
        runtime_profiles=("nvidia-vllm-k8s",)
        if vendor is AcceleratorVendor.NVIDIA
        else ("ascend-vllm-k8s",),
        capabilities=("bf16", "tensor-parallel"),
        accelerator_metadata_inferred=False,
    )


def _job(**overrides: object) -> Job:
    values: dict[str, Any] = {
        "id": "topology-fallback",
        "arrival_time": 0,
        "duration": 10,
        "gpu_count": 2,
        "gpu_memory_gb": 32,
        "allowed_vendors": (
            AcceleratorVendor.NVIDIA,
            AcceleratorVendor.HUAWEI_ASCEND,
        ),
        "allowed_kinds": (AcceleratorKind.GPU, AcceleratorKind.NPU),
        "required_capabilities": ("bf16",),
        "accelerator_request_explicit": True,
        "topology_mode": TopologyMode.REQUIRE_SAME_NODE,
    }
    values.update(overrides)
    return Job(**values)


def _topology_retry_cluster() -> Cluster:
    return Cluster(
        [
            Node(
                "ascend-a",
                [
                    _gpu(
                        "ascend-0",
                        "ascend-a",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    )
                ],
                topology={"zone": "z1", "rack": "r1"},
            ),
            Node(
                "ascend-b",
                [
                    _gpu(
                        "ascend-1",
                        "ascend-b",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    )
                ],
                topology={"zone": "z1", "rack": "r2"},
            ),
            Node(
                "nvidia-a",
                [
                    _gpu(
                        "nvidia-0",
                        "nvidia-a",
                        AcceleratorVendor.NVIDIA,
                        AcceleratorKind.GPU,
                    ),
                    _gpu(
                        "nvidia-1",
                        "nvidia-a",
                        AcceleratorVendor.NVIDIA,
                        AcceleratorKind.GPU,
                    ),
                ],
                topology={"zone": "z2", "rack": "r3"},
            ),
        ]
    )


@pytest.mark.parametrize(
    "scheduler_name",
    ["fifo", "binpack", "spread", "topology", "preemptive", "backfill"],
)
def test_topology_failure_on_first_vendor_retries_compatible_vendor(scheduler_name: str) -> None:
    cluster = _topology_retry_cluster()
    placement = create_scheduler(scheduler_name).place(cluster, _job())

    assert placement is not None
    assert len(placement) == 2
    assert {cluster.gpu_by_id(device_id).vendor for device_id in placement} == {
        AcceleratorVendor.NVIDIA
    }


def test_explicit_vendor_constraint_does_not_fallback_across_vendors() -> None:
    cluster = _topology_retry_cluster()
    job = _job(
        allowed_vendors=(AcceleratorVendor.HUAWEI_ASCEND,),
        allowed_kinds=(AcceleratorKind.NPU,),
    )

    assert create_scheduler("topology").place(cluster, job) is None


@pytest.mark.parametrize(
    ("outage_vendor", "healthy_vendor", "failed_id", "healthy_id"),
    [
        (
            AcceleratorVendor.HUAWEI_ASCEND,
            AcceleratorVendor.NVIDIA,
            "ascend-0",
            "nvidia-0",
        ),
        (
            AcceleratorVendor.NVIDIA,
            AcceleratorVendor.HUAWEI_ASCEND,
            "nvidia-0",
            "ascend-0",
        ),
    ],
)
def test_vendor_outage_preserves_other_vendor_on_mixed_node(
    outage_vendor: AcceleratorVendor,
    healthy_vendor: AcceleratorVendor,
    failed_id: str,
    healthy_id: str,
) -> None:
    scenario = Scenario(
        cluster=Cluster(
            [
                Node(
                    "mixed",
                    [
                        _gpu(
                            "nvidia-0",
                            "mixed",
                            AcceleratorVendor.NVIDIA,
                            AcceleratorKind.GPU,
                        ),
                        _gpu(
                            "ascend-0",
                            "mixed",
                            AcceleratorVendor.HUAWEI_ASCEND,
                            AcceleratorKind.NPU,
                        ),
                    ],
                    topology={"zone": "z1", "rack": "r1"},
                )
            ]
        ),
        jobs=[],
    )

    variant = _with_vendor_outage(scenario, outage_vendor)

    assert scenario.cluster.total_gpu_count == 2
    assert variant.cluster.total_gpu_count == 1
    assert variant.cluster.nodes[0].available
    assert variant.cluster.gpu_by_id(healthy_id).vendor is healthy_vendor
    with pytest.raises(KeyError):
        variant.cluster.gpu_by_id(failed_id)


def test_priority_preemption_does_not_evict_from_an_infeasible_vendor() -> None:
    cluster = Cluster(
        [
            Node(
                "nvidia",
                [
                    _gpu(
                        f"nvidia-{index}",
                        "nvidia",
                        AcceleratorVendor.NVIDIA,
                        AcceleratorKind.GPU,
                    )
                    for index in range(3)
                ],
            ),
            Node(
                "ascend",
                [
                    _gpu(
                        f"ascend-{index}",
                        "ascend",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    )
                    for index in range(2)
                ],
            ),
        ]
    )
    nvidia_victim = _job(
        id="nvidia-victim",
        gpu_count=2,
        allowed_vendors=(AcceleratorVendor.NVIDIA,),
        allowed_kinds=(AcceleratorKind.GPU,),
        priority=Priority.NORMAL,
    )
    ascend_victim = _job(
        id="ascend-victim",
        gpu_count=1,
        allowed_vendors=(AcceleratorVendor.HUAWEI_ASCEND,),
        allowed_kinds=(AcceleratorKind.NPU,),
        priority=Priority.LOW,
    )
    incoming = _job(
        id="incoming",
        arrival_time=1,
        gpu_count=3,
        priority=Priority.CRITICAL,
    )

    result = Simulator(
        cluster,
        [nvidia_victim, ascend_victim, incoming],
        PreemptiveScheduler(),
    ).run()
    simulated = {job.id: job for job in result.jobs}

    assert simulated["incoming"].first_start_time == 1
    assert simulated["nvidia-victim"].preemption_count == 1
    assert simulated["ascend-victim"].preemption_count == 0


def test_reclaim_keeps_victim_priority_ahead_of_vendor_disruption_cost() -> None:
    cluster = Cluster(
        [
            Node(
                "nvidia",
                [
                    _gpu(
                        "nvidia-0",
                        "nvidia",
                        AcceleratorVendor.NVIDIA,
                        AcceleratorKind.GPU,
                    )
                ],
            ),
            Node(
                "ascend",
                [
                    _gpu(
                        "ascend-0",
                        "ascend",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    )
                ],
            ),
        ]
    )
    scenario = Scenario(
        cluster,
        [
            _job(
                id="nvidia-victim",
                gpu_count=1,
                allowed_vendors=(AcceleratorVendor.NVIDIA,),
                allowed_kinds=(AcceleratorKind.GPU,),
                priority=Priority.NORMAL,
                queue_id="borrower",
            ),
            _job(
                id="ascend-victim",
                gpu_count=1,
                allowed_vendors=(AcceleratorVendor.HUAWEI_ASCEND,),
                allowed_kinds=(AcceleratorKind.NPU,),
                priority=Priority.LOW,
                checkpoint_cost=1,
                queue_id="borrower",
            ),
            _job(id="incoming", arrival_time=1, gpu_count=1, queue_id="product"),
        ],
        queues=(
            QueueSpec("borrower", "root", limit=ResourceVector(2, 128)),
            QueueSpec(
                "product",
                "root",
                guaranteed=ResourceVector(1, 64),
                limit=ResourceVector(1, 64),
            ),
        ),
    )

    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    simulated = {job.id: job for job in result.jobs}

    assert simulated["nvidia-victim"].reclaim_victim_count == 0
    assert simulated["ascend-victim"].reclaim_victim_count == 1


def test_reclaim_plan_keys_compare_elastic_and_victim_actions() -> None:
    elastic = _plan_reclaim_action_key(
        priority=int(Priority.NORMAL),
        action_rank=0,
        remaining_runtime=1.0,
        disruption_cost=1.0,
        job_id="elastic",
        gpu_id="nvidia-0",
    )
    victim = _plan_reclaim_action_key(
        priority=int(Priority.NORMAL),
        action_rank=1,
        remaining_runtime=1.0,
        disruption_cost=1.0,
        job_id="victim",
    )

    assert min((elastic, victim)) == elastic


def test_priority_preemption_keeps_victim_priority_ahead_of_vendor_cost() -> None:
    cluster = Cluster(
        [
            Node(
                "nvidia",
                [
                    _gpu(
                        "nvidia-0",
                        "nvidia",
                        AcceleratorVendor.NVIDIA,
                        AcceleratorKind.GPU,
                    )
                ],
            ),
            Node(
                "ascend",
                [
                    _gpu(
                        "ascend-0",
                        "ascend",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    )
                ],
            ),
        ]
    )
    higher_priority_free_victim = _job(
        id="nvidia-victim",
        gpu_count=1,
        allowed_vendors=(AcceleratorVendor.NVIDIA,),
        allowed_kinds=(AcceleratorKind.GPU,),
        priority=Priority.NORMAL,
    )
    lower_priority_costly_victim = _job(
        id="ascend-victim",
        gpu_count=1,
        allowed_vendors=(AcceleratorVendor.HUAWEI_ASCEND,),
        allowed_kinds=(AcceleratorKind.NPU,),
        priority=Priority.LOW,
        checkpoint_cost=1,
    )
    incoming = _job(
        id="incoming",
        arrival_time=1,
        gpu_count=1,
        priority=Priority.CRITICAL,
    )

    result = Simulator(
        cluster,
        [higher_priority_free_victim, lower_priority_costly_victim, incoming],
        PreemptiveScheduler(),
    ).run()
    simulated = {job.id: job for job in result.jobs}

    assert simulated["nvidia-victim"].preemption_count == 0
    assert simulated["ascend-victim"].preemption_count == 1


def test_elastic_runnable_demand_uses_largest_single_vendor_capacity() -> None:
    cluster = Cluster(
        [
            Node(
                "nvidia",
                [
                    _gpu(
                        f"nvidia-{index}",
                        "nvidia",
                        AcceleratorVendor.NVIDIA,
                        AcceleratorKind.GPU,
                    )
                    for index in range(2)
                ],
            ),
            Node(
                "ascend",
                [
                    _gpu(
                        f"ascend-{index}",
                        "ascend",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    )
                    for index in range(2)
                ],
            ),
        ]
    )
    scenario = Scenario(
        cluster,
        [
            _job(
                id="nvidia-blocker",
                duration=10,
                allowed_vendors=(AcceleratorVendor.NVIDIA,),
                allowed_kinds=(AcceleratorKind.GPU,),
                queue_id="blockers",
            ),
            _job(
                id="ascend-blocker",
                duration=10,
                allowed_vendors=(AcceleratorVendor.HUAWEI_ASCEND,),
                allowed_kinds=(AcceleratorKind.NPU,),
                queue_id="blockers",
            ),
            _job(
                id="elastic",
                arrival_time=1,
                duration=1,
                gpu_count=4,
                elastic=ElasticSpec(1, 4, 4),
                queue_id="elastic",
            ),
        ],
        queues=(
            QueueSpec("blockers", "root", limit=ResourceVector(4, 256)),
            QueueSpec("elastic", "root", guaranteed=ResourceVector(2, 128)),
        ),
    )

    result = Simulator.from_scenario(scenario, create_scheduler("drf", scenario)).run()

    satisfaction = result.metrics["queue_metrics"]["elastic"]["guaranteed_share_satisfaction"]
    assert satisfaction == pytest.approx(2 / 11)


def test_single_replica_elastic_resize_keeps_its_allocated_vendor() -> None:
    cluster = Cluster(
        [
            Node(
                "nvidia",
                [
                    _gpu(
                        "nvidia-0",
                        "nvidia",
                        AcceleratorVendor.NVIDIA,
                        AcceleratorKind.GPU,
                    )
                ],
            ),
            Node(
                "ascend",
                [
                    _gpu(
                        "ascend-0",
                        "ascend",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    )
                ],
            ),
        ]
    )
    job = _job(id="elastic", gpu_count=2, elastic=ElasticSpec(1, 2, 2))
    job.requested_replicas = 1
    cluster.allocate(job, ["nvidia-0"])
    job.current_replicas = 1

    assert cluster.eligible_gpus(job) == []


@pytest.mark.parametrize("mode", ["permissive", "quota-aware"])
def test_admission_rejects_mixed_vendor_gang_as_impossible(mode: str) -> None:
    cluster = Cluster(
        [
            Node(
                "mixed",
                [
                    _gpu(
                        "nvidia-0",
                        "mixed",
                        AcceleratorVendor.NVIDIA,
                        AcceleratorKind.GPU,
                    ),
                    _gpu(
                        "ascend-0",
                        "mixed",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    ),
                ],
            )
        ]
    )
    hierarchy = QueueHierarchy((QueueSpec("tenant", "root", limit=ResourceVector(2, 128)),))
    controller = AdmissionController(hierarchy, cluster, AccountingPolicy(), mode)

    decision = controller.decide(_job(id="mixed-gang", queue_id="tenant"))
    assert not decision.admitted
    assert decision.reason == "impossible_gpu_request"


def test_compare_default_excludes_vendor_preference_routes() -> None:
    args = build_parser().parse_args(["compare", "--scenario", "scenarios/demo.yaml"])

    assert args.schedulers == ",".join(DEFAULT_COMPARE_SCHEDULERS)
    assert "prefer-nvidia" not in args.schedulers
    assert "prefer-ascend" not in args.schedulers
    assert {"prefer-nvidia", "prefer-ascend"}.issubset(SCHEDULERS)


def test_reclaim_does_not_evict_from_an_infeasible_vendor() -> None:
    cluster = Cluster(
        [
            Node(
                "nvidia",
                [
                    _gpu(
                        f"nvidia-{index}",
                        "nvidia",
                        AcceleratorVendor.NVIDIA,
                        AcceleratorKind.GPU,
                    )
                    for index in range(3)
                ],
            ),
            Node(
                "ascend",
                [
                    _gpu(
                        f"ascend-{index}",
                        "ascend",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    )
                    for index in range(2)
                ],
            ),
        ]
    )
    scenario = Scenario(
        cluster,
        [
            _job(
                id="nvidia-victim",
                duration=10,
                gpu_count=2,
                allowed_vendors=(AcceleratorVendor.NVIDIA,),
                allowed_kinds=(AcceleratorKind.GPU,),
                queue_id="borrower",
            ),
            _job(
                id="ascend-victim",
                duration=10,
                gpu_count=1,
                allowed_vendors=(AcceleratorVendor.HUAWEI_ASCEND,),
                allowed_kinds=(AcceleratorKind.NPU,),
                priority=Priority.LOW,
                queue_id="borrower",
            ),
            _job(id="incoming", arrival_time=1, duration=1, gpu_count=3, queue_id="product"),
        ],
        queues=(
            QueueSpec("borrower", "root", limit=ResourceVector(3, 192)),
            QueueSpec(
                "product",
                "root",
                guaranteed=ResourceVector(3, 192),
                limit=ResourceVector(3, 192),
            ),
        ),
    )

    result = Simulator.from_scenario(
        scenario, create_scheduler("fairshare-reclaim", scenario)
    ).run()
    simulated = {job.id: job for job in result.jobs}

    assert simulated["incoming"].first_start_time == 1
    assert simulated["nvidia-victim"].preemption_count == 1
    assert simulated["ascend-victim"].preemption_count == 0
