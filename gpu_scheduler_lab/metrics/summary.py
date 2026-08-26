from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import mean, median
from typing import Any

from gpu_scheduler_lab.metrics.fairness import jains_fairness_index
from gpu_scheduler_lab.metrics.latency import percentile
from gpu_scheduler_lab.models.job import Job, JobStatus


def build_metrics(
    jobs: Sequence[Job],
    *,
    horizon: float,
    total_gpus: int,
    busy_gpu_time: float,
    memory_area: float,
    total_memory_gb: float,
    node_area: float,
    node_count: int,
    count_fragmentation_area: float,
    memory_fragmentation_area: float,
    peak_gpu_utilization: float,
    scheduling_attempts: int,
    failed_attempts: int,
    cross_node_gang_placements: int,
    same_node_gang_placements: int,
    same_rack_gang_placements: int,
    cross_rack_gang_placements: int,
    cross_zone_gang_placements: int,
    topology_distance_sum: float,
    topology_distance_samples: int,
    topology_requirement_violations: int,
) -> dict[str, Any]:
    completed = [job for job in jobs if job.status is JobStatus.COMPLETED]
    waits = [value for job in completed if (value := job.waiting_time) is not None]
    turnarounds = [value for job in completed if (value := job.turnaround_time) is not None]
    sla_jobs = [job for job in jobs if job.sla_deadline is not None]
    sla_violations = sum(
        job.completion_time is None or job.completion_time > job.sla_deadline  # type: ignore[operator]
        for job in sla_jobs
    )
    preemptions = sum(job.preemption_count for job in jobs)
    checkpoint_overhead = sum(job.checkpoint_overhead for job in jobs)
    restart_overhead = sum(job.restart_overhead for job in jobs)
    wasted_productive_gpu_time = sum(
        (job.checkpoint_overhead + job.restart_overhead) * job.gpu_count for job in jobs
    )

    demand_by_group: dict[str, float] = defaultdict(float)
    completed_by_group: dict[str, float] = defaultdict(float)
    turnaround_by_group: dict[str, float] = defaultdict(float)
    for job in jobs:
        group = job.group or job.priority.name.lower()
        demand_by_group[group] += job.duration * job.gpu_count
        if job.status is JobStatus.COMPLETED and job.turnaround_time is not None:
            completed_by_group[group] += job.duration * job.gpu_count
            turnaround_by_group[group] += job.turnaround_time * job.gpu_count
    fairness_groups: dict[str, dict[str, float]] = {}
    for group, demand in sorted(demand_by_group.items()):
        completed_gpu_time = completed_by_group[group]
        turnaround = turnaround_by_group[group]
        completion_ratio = completed_gpu_time / demand
        latency_efficiency = min(1.0, completed_gpu_time / turnaround) if turnaround else 0.0
        fairness_groups[group] = {
            "demand_gpu_time": demand,
            "completed_gpu_time": completed_gpu_time,
            "completion_ratio": completion_ratio,
            "turnaround_gpu_time": turnaround,
            "latency_efficiency": latency_efficiency,
            "service_quality": completion_ratio * latency_efficiency,
        }
    service_qualities = [group["service_quality"] for group in fairness_groups.values()]

    gpu_capacity_time = total_gpus * horizon
    memory_capacity_time = total_memory_gb * horizon
    node_capacity_time = node_count * horizon
    average_gpu_utilization = busy_gpu_time / gpu_capacity_time if gpu_capacity_time else 0.0
    return {
        "average_gpu_utilization": average_gpu_utilization,
        "peak_gpu_utilization": peak_gpu_utilization,
        "gpu_memory_utilization": memory_area / memory_capacity_time
        if memory_capacity_time
        else 0.0,
        "gpu_count_fragmentation": count_fragmentation_area / horizon if horizon else 0.0,
        "gpu_memory_fragmentation": memory_fragmentation_area / horizon if horizon else 0.0,
        "gpu_fragmentation_ratio": (
            (count_fragmentation_area + memory_fragmentation_area) / (2.0 * horizon)
            if horizon
            else 0.0
        ),
        "node_utilization": node_area / node_capacity_time if node_capacity_time else 0.0,
        "idle_gpu_time": max(0.0, gpu_capacity_time - busy_gpu_time),
        "average_waiting_time": mean(waits) if waits else 0.0,
        "median_waiting_time": median(waits) if waits else 0.0,
        "p95_waiting_time": percentile(waits, 0.95),
        "average_turnaround_time": mean(turnarounds) if turnarounds else 0.0,
        "p95_turnaround_time": percentile(turnarounds, 0.95),
        "completion_rate": len(completed) / len(jobs) if jobs else 1.0,
        "completed_jobs": len(completed),
        "total_jobs": len(jobs),
        "preemption_count": preemptions,
        "average_preemptions_per_job": preemptions / len(jobs) if jobs else 0.0,
        "sla_violation_count": sla_violations,
        "sla_violation_rate": sla_violations / len(sla_jobs) if sla_jobs else 0.0,
        "sla_job_count": len(sla_jobs),
        "cross_node_gang_placement_count": cross_node_gang_placements,
        "same_node_gang_placement_count": same_node_gang_placements,
        "same_rack_gang_placement_count": same_rack_gang_placements,
        "cross_rack_gang_placement_count": cross_rack_gang_placements,
        "cross_zone_gang_placement_count": cross_zone_gang_placements,
        "average_topology_distance": (
            topology_distance_sum / topology_distance_samples if topology_distance_samples else 0.0
        ),
        "topology_requirement_violation_count": topology_requirement_violations,
        "total_checkpoint_overhead": checkpoint_overhead,
        "total_restart_overhead": restart_overhead,
        "preemption_overhead_ratio": (
            wasted_productive_gpu_time / busy_gpu_time if busy_gpu_time else 0.0
        ),
        "wasted_productive_gpu_time": wasted_productive_gpu_time,
        "resumed_job_count": sum(job.preemption_count > 0 for job in completed),
        "scheduling_attempt_count": scheduling_attempts,
        "failed_placement_attempt_count": failed_attempts,
        "jains_fairness_index": jains_fairness_index(service_qualities),
        "fairness_groups": fairness_groups,
        "simulation_horizon": horizon,
        "busy_gpu_time": busy_gpu_time,
    }
