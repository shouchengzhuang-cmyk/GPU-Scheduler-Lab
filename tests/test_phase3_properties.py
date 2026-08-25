from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from gpu_scheduler_lab.fairshare.drf import weighted_dominant_share
from gpu_scheduler_lab.models import Job
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.queues import QueueSpec, ResourceVector
from gpu_scheduler_lab.scenario import Scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.schedulers.fifo import FIFOScheduler
from gpu_scheduler_lab.simulator.engine import Simulator


def _cluster(gpus: int) -> Cluster:
    return Cluster.from_dict(
        {
            "nodes": [
                {
                    "id": "node",
                    "gpus": [{"id": f"g{index}", "memory_gb": 40} for index in range(gpus)],
                }
            ]
        }
    )


@settings(max_examples=20, deadline=None)
@given(
    guarantee=st.integers(min_value=0, max_value=4),
    request=st.integers(min_value=1, max_value=4),
    arrivals=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=6),
)
def test_random_queue_runs_respect_hard_limits_and_are_deterministic(
    guarantee: int, request: int, arrivals: list[int]
) -> None:
    limit = 4
    scenario = Scenario(
        _cluster(limit),
        [
            Job(f"job-{index}", float(arrival), 2, request, 20, queue_id="tenant")
            for index, arrival in enumerate(arrivals)
        ],
        queues=(
            QueueSpec(
                "tenant",
                "root",
                guaranteed=ResourceVector(guarantee),
                limit=ResourceVector(limit, 160),
            ),
        ),
    )
    first = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    second = Simulator.from_scenario(scenario, create_scheduler("historical-drf", scenario)).run()
    assert first.metrics["queue_metrics"]["tenant"]["peak_gpu_usage"] <= limit
    assert [record.to_dict() for record in first.trace] == [
        record.to_dict() for record in second.trace
    ]
    json.dumps(first.to_dict(), allow_nan=False)


@settings(max_examples=30, deadline=None)
@given(
    gpu_usage=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    memory_usage=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    weight=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
)
def test_drf_property_stays_finite(gpu_usage: float, memory_usage: float, weight: float) -> None:
    value = weighted_dominant_share(
        ResourceVector(gpu_usage, memory_usage), ResourceVector(100, 1000), weight
    )
    assert value >= 0
    assert value < float("inf")


def test_full_cluster_short_circuit_preserves_completion_without_quadratic_attempts() -> None:
    jobs = [Job(f"job-{index:03d}", 0, 1, 1, 20) for index in range(100)]
    result = Simulator(_cluster(1), jobs, FIFOScheduler()).run()
    assert result.metrics["completed_jobs"] == 100
    assert result.metrics["scheduling_attempt_count"] == 100
