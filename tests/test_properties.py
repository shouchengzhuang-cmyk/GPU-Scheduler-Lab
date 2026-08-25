from __future__ import annotations

from conftest import make_cluster
from hypothesis import given, settings
from hypothesis import strategies as st

from gpu_scheduler_lab.models import EventType, Job
from gpu_scheduler_lab.schedulers import FIFOScheduler
from gpu_scheduler_lab.simulator.engine import Simulator


@settings(max_examples=30, deadline=None)
@given(
    capacities=st.lists(st.integers(min_value=8, max_value=80), min_size=1, max_size=6),
    requests=st.lists(st.integers(min_value=1, max_value=80), min_size=0, max_size=12),
)
def test_resource_gang_completion_and_conservation_invariants(
    capacities: list[int], requests: list[int]
) -> None:
    cluster = make_cluster([list(map(float, capacities))])
    jobs = [
        Job(
            id=f"job-{index}",
            arrival_time=float(index % 3),
            duration=float(index % 5 + 1),
            gpu_count=1,
            gpu_memory_gb=float(memory),
            gang=True,
        )
        for index, memory in enumerate(requests)
    ]

    result = Simulator(cluster, jobs, FIFOScheduler()).run()

    for job in result.jobs:
        assert len(job.allocated_gpu_ids) in {0, job.gpu_count}
        if job.completion_time is not None:
            assert job.allocated_gpu_ids == []
    horizon = float(result.metrics["simulation_horizon"])
    assert (
        result.metrics["busy_gpu_time"] + result.metrics["idle_gpu_time"]
        == len(capacities) * horizon
    )


@settings(max_examples=25, deadline=None)
@given(
    arrivals=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=10),
    durations=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10),
)
def test_determinism_invariant(arrivals: list[int], durations: list[int]) -> None:
    size = min(len(arrivals), len(durations))
    jobs = [Job(f"job-{index}", arrivals[index], durations[index], 1, 10) for index in range(size)]
    cluster = make_cluster([[24, 24]])

    first = Simulator(cluster, jobs, FIFOScheduler()).run()
    second = Simulator(cluster, jobs, FIFOScheduler()).run()

    assert first.trace == second.trace
    assert first.metrics == second.metrics


def test_ownership_invariant_from_trace() -> None:
    jobs = [Job(f"job-{index}", 0, 5 + index, 1, 10) for index in range(6)]
    result = Simulator(make_cluster([[24, 24]]), jobs, FIFOScheduler()).run()
    owners: dict[str, str] = {}

    for record in result.trace:
        if record.event in {EventType.JOB_START, EventType.JOB_RESUME}:
            for gpu_id in record.gpu_ids:
                assert gpu_id not in owners
                owners[gpu_id] = record.job_id
        elif record.event in {EventType.JOB_COMPLETE, EventType.JOB_PREEMPT}:
            for gpu_id in record.gpu_ids:
                assert owners.pop(gpu_id) == record.job_id
    assert owners == {}
