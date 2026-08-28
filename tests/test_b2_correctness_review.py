from __future__ import annotations

from typing import Any

import pytest

from gpu_scheduler_lab.heterogeneous.study import _with_vendor_outage
from gpu_scheduler_lab.models import (
    GPU,
    AcceleratorKind,
    AcceleratorVendor,
    Cluster,
    Job,
    Node,
    TopologyMode,
)
from gpu_scheduler_lab.scenario import Scenario
from gpu_scheduler_lab.schedulers import create_scheduler


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
def test_topology_failure_on_first_vendor_retries_compatible_vendor(
    scheduler_name: str,
) -> None:
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
