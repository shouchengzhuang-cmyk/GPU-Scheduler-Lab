from __future__ import annotations

from conftest import make_cluster
from hypothesis import given, settings
from hypothesis import strategies as st

from gpu_scheduler_lab.experiments import scenario_hash
from gpu_scheduler_lab.models import GPU, Cluster, EventType, Job, Node, Priority, TopologyMode
from gpu_scheduler_lab.scenario import Scenario
from gpu_scheduler_lab.schedulers import (
    FIFOScheduler,
    PreemptiveScheduler,
    ReservationBackfillScheduler,
    TopologyAwareScheduler,
)
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


@settings(max_examples=20, deadline=None)
@given(gpu_count=st.integers(min_value=2, max_value=4))
def test_required_topology_is_never_violated(gpu_count: int) -> None:
    cluster = Cluster(
        [
            Node(
                id=f"node-{index}",
                topology={"zone": "z1", "rack": f"r{index // 2}"},
                gpus=[GPU(f"gpu-{index}", f"node-{index}", 24)],
            )
            for index in range(4)
        ]
    )
    job = Job(
        "gang",
        0,
        2,
        gpu_count,
        20,
        gang=True,
        topology_mode=TopologyMode.REQUIRE_SAME_RACK,
    )

    placement = TopologyAwareScheduler().place(cluster, job)

    if placement is not None:
        racks = {
            cluster.nodes[int(cluster.gpu_by_id(gpu_id).node_id.rsplit("-", 1)[1])].topology["rack"]
            for gpu_id in placement
        }
        assert len(racks) == 1


@settings(max_examples=20, deadline=None)
@given(blocker_duration=st.integers(min_value=4, max_value=30))
def test_backfill_never_delays_reservation_guarantee(blocker_duration: int) -> None:
    jobs = [
        Job("a-blocker", 0, blocker_duration, 3, 20, gang=True),
        Job("b-reserved", 1, 10, 4, 20, gang=True),
        Job("c-short", 1, max(1, blocker_duration - 2), 1, 20),
    ]

    result = Simulator(make_cluster([[24, 24, 24, 24]]), jobs, ReservationBackfillScheduler()).run()

    reserved = next(job for job in result.jobs if job.id == "b-reserved")
    assert reserved.first_start_time == blocker_duration
    assert result.metrics["reservation_delay_violation_count"] == 0


@settings(max_examples=20, deadline=None)
@given(
    checkpoint_cost=st.integers(min_value=0, max_value=8),
    restart_cost=st.integers(min_value=0, max_value=8),
)
def test_preemption_productive_runtime_conservation(
    checkpoint_cost: int, restart_cost: int
) -> None:
    jobs = [
        Job(
            "low",
            0,
            20,
            1,
            20,
            priority=Priority.LOW,
            checkpoint_cost=checkpoint_cost,
            restart_cost=restart_cost,
        ),
        Job("critical", 5, 2, 1, 20, priority=Priority.CRITICAL),
    ]

    result = Simulator(make_cluster([[24]]), jobs, PreemptiveScheduler()).run()
    low = next(job for job in result.jobs if job.id == "low")

    assert low.accumulated_runtime == low.duration
    assert result.metrics["wasted_productive_gpu_time"] == checkpoint_cost + restart_cost


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1000))
def test_scenario_hash_and_phase2_determinism(seed: int) -> None:
    scenario = Scenario(
        make_cluster([[24, 80], [40, 80]]),
        [Job("job", seed % 3, 5, 2, 20, gang=True)],
        metadata={"seed": seed},
    )
    first = Simulator(scenario.cluster, scenario.jobs, TopologyAwareScheduler()).run()
    second = Simulator(scenario.cluster, scenario.jobs, TopologyAwareScheduler()).run()

    assert first.trace == second.trace
    assert first.metrics == second.metrics
    assert scenario_hash(scenario) == scenario_hash(scenario.clone())
