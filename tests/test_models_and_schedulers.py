from __future__ import annotations

import pytest
from conftest import make_cluster

from gpu_scheduler_lab.models import Job
from gpu_scheduler_lab.schedulers import BinPackScheduler, FIFOScheduler, SpreadScheduler


def test_fifo_rejects_insufficient_memory_without_mutation() -> None:
    cluster = make_cluster([[24, 24]])
    job = Job("large", 0, 10, 1, 40)

    assert FIFOScheduler().place(cluster, job) is None
    assert all(gpu.free for gpu in cluster.gpus)


def test_fifo_rejects_insufficient_gpu_count() -> None:
    cluster = make_cluster([[80, 80]])
    job = Job("gang", 0, 10, 3, 40, gang=True)

    assert FIFOScheduler().place(cluster, job) is None


def test_fifo_handles_heterogeneous_gpus_deterministically() -> None:
    cluster = make_cluster([[24, 80], [40, 80]])
    job = Job("job", 0, 10, 2, 40)

    assert FIFOScheduler().place(cluster, job) == ["gpu-0-1", "gpu-1-0"]
    assert FIFOScheduler().place(cluster, job) == ["gpu-0-1", "gpu-1-0"]


def test_cluster_allocation_is_atomic_and_releases_resources() -> None:
    cluster = make_cluster([[24, 24]])
    job = Job("job", 0, 10, 2, 20, gang=True)

    with pytest.raises(ValueError, match="exactly"):
        cluster.allocate(job, ["gpu-0-0"])
    assert all(gpu.free for gpu in cluster.gpus)

    cluster.allocate(job, ["gpu-0-0", "gpu-0-1"])
    assert all(gpu.owner_job_id == "job" for gpu in cluster.gpus)
    cluster.release(job)
    assert all(gpu.free and gpu.allocated_memory_gb == 0 for gpu in cluster.gpus)


def test_oversized_allocation_never_corrupts_cluster() -> None:
    cluster = make_cluster([[24]])
    job = Job("job", 0, 10, 1, 80)

    with pytest.raises(ValueError, match="undersized"):
        cluster.allocate(job, ["gpu-0-0"])
    cluster.assert_invariants()
    assert cluster.gpus[0].free


def test_binpack_prefers_partially_used_node() -> None:
    cluster = make_cluster([[24, 24], [24, 24]])
    existing = Job("existing", 0, 20, 1, 20)
    cluster.allocate(existing, ["gpu-1-0"])

    placement = BinPackScheduler().place(cluster, Job("new", 1, 10, 1, 20))

    assert placement == ["gpu-1-1"]


def test_binpack_prefers_tight_memory_fit() -> None:
    cluster = make_cluster([[80, 24]])

    placement = BinPackScheduler().place(cluster, Job("new", 0, 10, 1, 20))

    assert placement == ["gpu-0-1"]


def test_spread_uses_distinct_nodes_before_second_gpu_on_a_node() -> None:
    cluster = make_cluster([[24, 24], [24, 24]])

    placement = SpreadScheduler().place(cluster, Job("job", 0, 10, 2, 20, gang=True))

    assert placement == ["gpu-0-0", "gpu-1-0"]
