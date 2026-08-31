from __future__ import annotations

from typing import Any

import pytest

from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.models import (
    GPU,
    AcceleratorKind,
    AcceleratorVendor,
    Cluster,
    Job,
    Node,
)
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


def _cluster(*, nvidia_available: bool = True, ascend_available: bool = True) -> Cluster:
    return Cluster(
        [
            Node(
                "nvidia-a",
                [
                    _gpu("device-0", "nvidia-a", AcceleratorVendor.NVIDIA, AcceleratorKind.GPU),
                    _gpu("device-2", "nvidia-a", AcceleratorVendor.NVIDIA, AcceleratorKind.GPU),
                ],
                available=nvidia_available,
                topology={"zone": "z1", "rack": "r1"},
            ),
            Node(
                "ascend-a",
                [
                    _gpu(
                        "device-1",
                        "ascend-a",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    ),
                    _gpu(
                        "device-3",
                        "ascend-a",
                        AcceleratorVendor.HUAWEI_ASCEND,
                        AcceleratorKind.NPU,
                    ),
                ],
                available=ascend_available,
                topology={"zone": "z2", "rack": "r2"},
            ),
        ]
    )


def _job(**overrides: object) -> Job:
    values: dict[str, Any] = {
        "id": "dual-stack-job",
        "arrival_time": 0,
        "duration": 10,
        "gpu_count": 1,
        "gpu_memory_gb": 32,
        "allowed_vendors": (
            AcceleratorVendor.NVIDIA,
            AcceleratorVendor.HUAWEI_ASCEND,
        ),
        "allowed_kinds": (AcceleratorKind.GPU, AcceleratorKind.NPU),
        "required_capabilities": ("bf16",),
        "accelerator_request_explicit": True,
    }
    values.update(overrides)
    return Job(**values)


def test_typed_device_compatibility_checks_all_accelerator_constraints() -> None:
    cluster = _cluster()
    nvidia = cluster.gpu_by_id("device-0")
    ascend = cluster.gpu_by_id("device-1")
    request = _job(
        allowed_vendors=(AcceleratorVendor.NVIDIA,),
        allowed_kinds=(AcceleratorKind.GPU,),
        allowed_models=("A100-80GB",),
        runtime_profile="nvidia-vllm-k8s",
    )

    assert nvidia.is_compatible(request)
    assert not ascend.is_compatible(request)
    assert not nvidia.is_compatible(_job(required_capabilities=("fp8",)))
    assert not nvidia.is_compatible(_job(runtime_profile="ascend-vllm-k8s"))
    assert not nvidia.is_compatible(_job(allowed_models=("Ascend-910B",)))


@pytest.mark.parametrize(
    "scheduler_name",
    ["fifo", "binpack", "spread", "topology", "preemptive", "backfill"],
)
def test_all_placement_schedulers_keep_gang_on_one_vendor(scheduler_name: str) -> None:
    cluster = _cluster()
    job = _job(gpu_count=2)

    placement = create_scheduler(scheduler_name).place(cluster, job)

    assert placement is not None
    assert len({cluster.gpu_by_id(device_id).vendor for device_id in placement}) == 1
    cluster.allocate(job, placement)
    cluster.assert_invariants()


def test_cluster_rejects_cross_vendor_gang_even_if_scheduler_is_wrong() -> None:
    cluster = _cluster()

    with pytest.raises(ValueError, match="must not mix accelerator vendors"):
        cluster.allocate(_job(gpu_count=2), ["device-0", "device-1"])


def test_vendor_preference_uses_preferred_vendor_and_bounded_fallback() -> None:
    job = _job()
    available = _cluster()
    outage = _cluster(nvidia_available=False)

    preferred = create_scheduler("prefer-nvidia").place(available, job)
    fallback = create_scheduler("prefer-nvidia").place(outage, job)

    assert preferred is not None
    assert {available.gpu_by_id(device_id).vendor for device_id in preferred} == {
        AcceleratorVendor.NVIDIA
    }
    assert fallback is not None
    assert {outage.gpu_by_id(device_id).vendor for device_id in fallback} == {
        AcceleratorVendor.HUAWEI_ASCEND
    }


def test_internal_job_contract_rejects_impossible_vendor_kind_pair() -> None:
    with pytest.raises(ValueError, match="must include a supported pair"):
        _job(
            allowed_vendors=(AcceleratorVendor.NVIDIA,),
            allowed_kinds=(AcceleratorKind.NPU,),
        )


def test_device_contract_rejects_known_vendor_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="supported accelerator pair"):
        _gpu(
            "invalid",
            "node",
            AcceleratorVendor.NVIDIA,
            AcceleratorKind.NPU,
        )


def test_accounting_demand_does_not_combine_vendors_for_a_gang() -> None:
    with pytest.raises(ValueError, match="insufficient compatible GPUs"):
        AccountingPolicy().minimum_demand(_job(gpu_count=3), _cluster().gpus, 3)
