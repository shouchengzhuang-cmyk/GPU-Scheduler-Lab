from __future__ import annotations

import pytest
from conftest import make_cluster

from gpu_scheduler_lab.elastic import ElasticSpec
from gpu_scheduler_lab.metrics import fragmentation_snapshot, jains_fairness_index
from gpu_scheduler_lab.models import Job, JobStatus


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


def test_unschedulable_nodes_do_not_dilute_fragmentation() -> None:
    cluster = make_cluster([[24, 24], [24, 24, 24, 24, 24, 24, 24, 24]])
    cluster.nodes[1].schedulable = False
    cluster.allocate(Job("a", 0, 10, 1, 20), ["gpu-0-0"])

    count, memory, combined = fragmentation_snapshot(cluster)

    assert count == 1.0
    assert memory == pytest.approx(1 - 20 / 24)
    assert combined == pytest.approx((count + memory) / 2)


def test_jains_fairness_index() -> None:
    assert jains_fairness_index([1, 1, 1]) == 1.0
    assert jains_fairness_index([1, 0]) == 0.5
    assert jains_fairness_index([]) == 1.0


def test_waiting_time_is_submission_to_first_start_for_fixed_and_elastic_jobs() -> None:
    fixed = Job("fixed", 1, 10, 1, 20)
    elastic = Job(
        "elastic",
        1,
        10,
        2,
        20,
        elastic=ElasticSpec(min_replicas=1, preferred_replicas=2, max_replicas=4),
    )

    for job, completion_time in ((fixed, 18.0), (elastic, 30.0)):
        job.first_start_time = 4.0
        job.completion_time = completion_time
        job.status = JobStatus.COMPLETED

    assert fixed.waiting_time == pytest.approx(3.0)
    assert elastic.waiting_time == pytest.approx(3.0)
