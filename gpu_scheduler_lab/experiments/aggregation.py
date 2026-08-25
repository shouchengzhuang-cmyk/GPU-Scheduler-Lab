from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any

from gpu_scheduler_lab.metrics.latency import percentile

DEFAULT_METRICS = (
    "average_gpu_utilization",
    "gpu_fragmentation_ratio",
    "average_waiting_time",
    "sla_violation_rate",
    "jains_fairness_index",
    "average_topology_distance",
    "preemption_overhead_ratio",
    "average_queue_wait_time",
    "queue_service_jains_index",
    "guarantee_satisfaction_variance",
    "starvation_count",
    "elastic_scale_up_count",
    "elastic_scale_down_count",
    "recovery_overhead",
)


def aggregate_runs(
    runs: list[dict[str, Any]], metrics: tuple[str, ...] = DEFAULT_METRICS
) -> list[dict[str, str | int | float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in runs:
        scheduler = str(run["scheduler"])
        run_metrics = run["metrics"]
        if not isinstance(run_metrics, dict):
            raise ValueError("run metrics must be a mapping")
        for metric in metrics:
            value = run_metrics.get(metric)
            if isinstance(value, int | float) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    values[(scheduler, metric)].append(numeric)
    rows: list[dict[str, str | int | float]] = []
    for (scheduler, metric), samples in sorted(values.items()):
        rows.append(
            {
                "scheduler": scheduler,
                "metric": metric,
                "runs": len(samples),
                "mean": mean(samples),
                "stddev": pstdev(samples),
                "median": median(samples),
                "p95": percentile(samples, 0.95),
            }
        )
    return rows
