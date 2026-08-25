from __future__ import annotations

from conftest import make_cluster

from gpu_scheduler_lab.models import EventType, Job, JobStatus, Priority
from gpu_scheduler_lab.schedulers import FIFOScheduler, PreemptiveScheduler, SpreadScheduler
from gpu_scheduler_lab.simulator.engine import SimulationResult, Simulator


def _job(result: SimulationResult, job_id: str) -> Job:
    return next(job for job in result.jobs if job.id == job_id)


def test_simultaneous_arrivals_use_stable_job_id_order() -> None:
    jobs = [Job("b", 0, 5, 1, 10), Job("a", 0, 5, 1, 10)]

    result = Simulator(make_cluster([[24]]), jobs, FIFOScheduler()).run()
    starts = [record.job_id for record in result.trace if record.event is EventType.JOB_START]

    assert starts == ["a", "b"]


def test_completion_precedes_arrival_at_same_time() -> None:
    jobs = [Job("first", 0, 10, 1, 10), Job("second", 10, 5, 1, 10)]

    result = Simulator(make_cluster([[24]]), jobs, FIFOScheduler()).run()
    second = _job(result, "second")

    assert second.first_start_time == 10
    at_ten = [record.event for record in result.trace if record.time == 10]
    assert at_ten == [EventType.JOB_COMPLETE, EventType.JOB_ARRIVAL, EventType.JOB_START]


def test_preemption_resumes_only_remaining_runtime() -> None:
    jobs = [
        Job("low", 0, 100, 1, 20, priority=Priority.LOW),
        Job("critical", 10, 10, 1, 20, priority=Priority.CRITICAL),
    ]

    result = Simulator(make_cluster([[80]]), jobs, PreemptiveScheduler()).run()
    low = _job(result, "low")

    assert low.preemption_count == 1
    assert low.accumulated_runtime == 100
    assert low.completion_time == 110
    assert any(
        record.event is EventType.JOB_RESUME and record.job_id == "low" for record in result.trace
    )


def test_equal_priority_job_cannot_preempt() -> None:
    jobs = [
        Job("first", 0, 100, 1, 20, priority=Priority.NORMAL),
        Job("second", 1, 10, 1, 20, priority=Priority.NORMAL),
    ]

    result = Simulator(make_cluster([[24]]), jobs, PreemptiveScheduler()).run()

    assert _job(result, "first").preemption_count == 0
    assert _job(result, "second").first_start_time == 100


def test_aging_protects_waiting_low_priority_job() -> None:
    jobs = [
        Job("high-1", 0, 40, 1, 20, priority=Priority.HIGH),
        Job("low", 0, 10, 1, 20, priority=Priority.LOW),
        Job("high-2", 20, 40, 1, 20, priority=Priority.HIGH),
        Job("high-3", 75, 40, 1, 20, priority=Priority.HIGH),
    ]

    result = Simulator(make_cluster([[24]]), jobs, PreemptiveScheduler()).run()
    low = _job(result, "low")

    assert low.first_start_time == 80
    assert all(job.preemption_count == 0 for job in result.jobs)


def test_preemption_does_not_evict_jobs_on_unsuitable_gpus() -> None:
    jobs = [
        Job("small", 0, 100, 1, 20, priority=Priority.LOW),
        Job("large", 1, 5, 1, 70, priority=Priority.CRITICAL),
    ]

    result = Simulator(make_cluster([[24], [80]]), jobs, PreemptiveScheduler()).run()

    assert _job(result, "small").preemption_count == 0
    assert _job(result, "large").first_start_time == 1


def test_cross_node_gang_placement_is_counted() -> None:
    job = Job("gang", 0, 10, 2, 20, gang=True)

    result = Simulator(make_cluster([[24], [24]]), [job], SpreadScheduler()).run()

    assert result.metrics["cross_node_gang_placement_count"] == 1


def test_sla_violation_and_utilization_metrics() -> None:
    jobs = [
        Job("on-time", 0, 10, 1, 20, sla_deadline=10),
        Job("late", 0, 10, 1, 20, sla_deadline=15),
    ]

    result = Simulator(make_cluster([[24]]), jobs, FIFOScheduler()).run()

    assert result.metrics["sla_violation_count"] == 1
    assert result.metrics["sla_violation_rate"] == 0.5
    assert result.metrics["average_gpu_utilization"] == 1.0
    assert result.metrics["idle_gpu_time"] == 0.0


def test_zero_job_scenario_is_well_defined() -> None:
    result = Simulator(make_cluster([[24]]), [], FIFOScheduler()).run()

    assert result.metrics["completion_rate"] == 1.0
    assert result.metrics["simulation_horizon"] == 0.0
    assert result.trace == []


def test_impossible_job_remains_pending_without_allocations() -> None:
    result = Simulator(
        make_cluster([[24]]), [Job("impossible", 0, 10, 2, 20, gang=True)], FIFOScheduler()
    ).run()
    job = _job(result, "impossible")

    assert job.status is JobStatus.PENDING
    assert job.allocated_gpu_ids == []
    assert result.metrics["completion_rate"] == 0.0


def test_same_input_has_identical_trace_and_metrics() -> None:
    cluster = make_cluster([[24, 40], [80, 80]])
    jobs = [
        Job("a", 0, 10, 1, 20),
        Job("b", 1, 15, 2, 40, gang=True),
    ]

    first = Simulator(cluster, jobs, FIFOScheduler()).run()
    second = Simulator(cluster, jobs, FIFOScheduler()).run()

    assert first.trace == second.trace
    assert first.metrics == second.metrics
    assert first.to_dict()["jobs"] == second.to_dict()["jobs"]
