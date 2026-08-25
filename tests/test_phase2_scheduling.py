from __future__ import annotations

from pathlib import Path

from conftest import make_cluster

from gpu_scheduler_lab.models import EventType, Job, JobStatus, Priority
from gpu_scheduler_lab.scenario import load_scenario
from gpu_scheduler_lab.schedulers import (
    FIFOScheduler,
    PreemptiveScheduler,
    ReservationBackfillScheduler,
    SpreadScheduler,
    TopologyAwareScheduler,
)
from gpu_scheduler_lab.simulator.engine import SimulationResult, Simulator


def _job(result: SimulationResult, job_id: str) -> Job:
    return next(job for job in result.jobs if job.id == job_id)


def test_topology_scenario_reduces_distance_without_violation() -> None:
    scenario = load_scenario(Path("scenarios/topology.yaml"))
    spread = Simulator(scenario.cluster, scenario.jobs, SpreadScheduler()).run()
    topology = Simulator(scenario.cluster, scenario.jobs, TopologyAwareScheduler()).run()

    assert (
        topology.metrics["average_topology_distance"] < spread.metrics["average_topology_distance"]
    )
    assert topology.metrics["same_rack_gang_placement_count"] == 1
    assert topology.metrics["topology_requirement_violation_count"] == 0


def test_backfill_honors_reservation_and_prioritizes_short_fit() -> None:
    scenario = load_scenario(Path("scenarios/backfill.yaml"))
    fifo = Simulator(scenario.cluster, scenario.jobs, FIFOScheduler()).run()
    backfill = Simulator(scenario.cluster, scenario.jobs, ReservationBackfillScheduler()).run()

    assert _job(backfill, "b-reserved").first_start_time == 10
    assert _job(fifo, "b-reserved").first_start_time == 11
    assert _job(backfill, "d-short-small").first_start_time == 1
    assert _job(fifo, "d-short-small").first_start_time == 9
    assert backfill.metrics["reservation_count"] == 1
    assert backfill.metrics["successful_backfill_count"] == 1
    assert backfill.metrics["reservation_delay_violation_count"] == 0


def test_impossible_reservation_does_not_block_feasible_job() -> None:
    jobs = [
        Job("impossible", 0, 10, 3, 20, gang=True),
        Job("small", 0, 2, 1, 20),
    ]

    result = Simulator(make_cluster([[24, 24]]), jobs, ReservationBackfillScheduler()).run()

    assert _job(result, "small").completion_time == 2
    assert result.metrics["reservation_count"] == 0


def test_checkpoint_and_restart_costs_hold_resources_without_productive_progress() -> None:
    scenario = load_scenario(Path("scenarios/preemption-cost.yaml"))
    result = Simulator(scenario.cluster, scenario.jobs, PreemptiveScheduler()).run()
    low = _job(result, "low")
    events = [(record.time, record.event) for record in result.trace]

    assert (5.0, EventType.JOB_PREEMPT) in events
    assert (85.0, EventType.JOB_CHECKPOINT_COMPLETE) in events
    assert (95.0, EventType.JOB_RESTART) in events
    assert (175.0, EventType.JOB_RESTART_COMPLETE) in events
    assert low.accumulated_runtime == 100
    assert low.completion_time == 270
    assert result.metrics["busy_gpu_time"] == 270
    assert result.metrics["total_checkpoint_overhead"] == 80
    assert result.metrics["total_restart_overhead"] == 80
    assert result.metrics["wasted_productive_gpu_time"] == 160


def test_checkpoint_victims_stay_suspended_until_incoming_job_starts() -> None:
    cluster = make_cluster([[24], [24]])
    jobs = [
        Job("victim-a", 0, 20, 1, 20, Priority.LOW, checkpoint_cost=1),
        Job("victim-b", 0, 20, 1, 20, Priority.LOW, checkpoint_cost=2),
        Job("incoming", 1, 1, 2, 20, Priority.CRITICAL, gang=True),
        Job("other", 1.5, 1, 1, 20, Priority.NORMAL),
    ]

    result = Simulator(cluster, jobs, PreemptiveScheduler()).run()

    assert _job(result, "incoming").first_start_time == 3
    assert _job(result, "other").first_start_time == 4
    assert _job(result, "victim-a").preemption_count == 1
    assert _job(result, "victim-b").preemption_count == 1
    assert result.metrics["preemption_count"] == 2
    assert all(job.status is JobStatus.COMPLETED for job in result.jobs)


def test_zero_cost_preemption_keeps_existing_resume_semantics() -> None:
    jobs = [
        Job("low", 0, 100, 1, 20, priority=Priority.LOW),
        Job("critical", 5, 10, 1, 20, priority=Priority.CRITICAL),
    ]

    result = Simulator(make_cluster([[24]]), jobs, PreemptiveScheduler()).run()

    assert _job(result, "critical").first_start_time == 5
    assert _job(result, "low").completion_time == 110
    assert any(record.event is EventType.JOB_RESUME for record in result.trace)
    assert not any(record.event is EventType.JOB_CHECKPOINT_COMPLETE for record in result.trace)


def test_high_preemption_cost_can_be_worse_than_fifo() -> None:
    scenario = load_scenario(Path("scenarios/preemption-cost.yaml"))
    fifo = Simulator(scenario.cluster, scenario.jobs, FIFOScheduler()).run()
    preemptive = Simulator(scenario.cluster, scenario.jobs, PreemptiveScheduler()).run()

    assert preemptive.metrics["average_waiting_time"] > fifo.metrics["average_waiting_time"]
    assert preemptive.metrics["preemption_overhead_ratio"] > 0
