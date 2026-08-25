from __future__ import annotations

import pytest
from conftest import make_cluster

from gpu_scheduler_lab.metrics import fragmentation_snapshot, jains_fairness_index
from gpu_scheduler_lab.models import Job


def test_count_fragmentation_bounds_and_meaning() -> None:
    cluster = make_cluster([[24, 24], [24, 24]])
    first = Job("a", 0, 10, 1, 20)
    second = Job("b", 0, 10, 1, 20)
    cluster.allocate(first, ["gpu-0-0"])
    cluster.allocate(second, ["gpu-1-0"])

    count, _, _ = fragmentation_snapshot(cluster)

    assert count == 1.0


def test_full_and_empty_nodes_have_zero_count_fragmentation() -> None:
    cluster = make_cluster([[24, 24], [24, 24]])
    job = Job("a", 0, 10, 2, 20)
    cluster.allocate(job, ["gpu-0-0", "gpu-0-1"])

    count, _, _ = fragmentation_snapshot(cluster)

    assert count == 0.0


def test_memory_fragmentation_measures_stranded_memory() -> None:
    cluster = make_cluster([[80]])
    job = Job("a", 0, 10, 1, 20)
    cluster.allocate(job, ["gpu-0-0"])

    _, memory, combined = fragmentation_snapshot(cluster)

    assert memory == pytest.approx(0.75)
    assert combined == pytest.approx(0.375)


def test_jains_fairness_index() -> None:
    assert jains_fairness_index([1, 1, 1]) == 1.0
    assert jains_fairness_index([1, 0]) == 0.5
    assert jains_fairness_index([]) == 1.0
