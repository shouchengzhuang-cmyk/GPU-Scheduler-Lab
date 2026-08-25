from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_phase3_timelines(runs: list[dict[str, Any]], output: Path) -> None:
    if not runs:
        return
    metrics = runs[0].get("metrics", {})
    if not isinstance(metrics, dict):
        return
    _plot_queue_share(metrics, output / "queue-share-timeline.png")
    _plot_queue_value(
        metrics,
        "borrowed_usage",
        "Borrowed GPU units",
        output / "borrowed-capacity-timeline.png",
    )
    _plot_queue_value(
        metrics,
        "fairshare_debt",
        "Fair-share debt",
        output / "fairshare-debt-timeline.png",
    )
    _plot_mapping_timeline(
        metrics.get("elastic_replica_timeline", []),
        "replicas",
        "Allocated replicas",
        output / "elastic-replica-timeline.png",
    )
    _plot_fleet(metrics, output / "fleet-capacity-timeline.png")


def _plot_queue_share(metrics: dict[str, Any], path: Path) -> None:
    timeline = metrics.get("queue_timeline", [])
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    queue_ids = _queue_ids(timeline)
    for queue_id in queue_ids:
        axis.step(
            [float(point["time"]) for point in timeline],
            [float(point["queues"].get(queue_id, {}).get("gpu_units", 0.0)) for point in timeline],
            where="post",
            label=queue_id,
        )
        guarantee = metrics.get("queue_metrics", {}).get(queue_id, {}).get("guaranteed_gpu_units")
        if isinstance(guarantee, int | float) and guarantee > 0:
            axis.axhline(float(guarantee), linestyle="--", alpha=0.35)
    _finish(axis, "Queue GPU share", "GPU units", queue_ids)
    _save(figure, path)


def _plot_queue_value(metrics: dict[str, Any], field: str, ylabel: str, path: Path) -> None:
    timeline = metrics.get("queue_timeline", [])
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    queue_ids = _queue_ids(timeline)
    for queue_id in queue_ids:
        axis.step(
            [float(point["time"]) for point in timeline],
            [float(point["queues"].get(queue_id, {}).get(field, 0.0)) for point in timeline],
            where="post",
            label=queue_id,
        )
    _finish(axis, ylabel, ylabel, queue_ids)
    _save(figure, path)


def _plot_mapping_timeline(raw: Any, field: str, ylabel: str, path: Path) -> None:
    timeline = raw if isinstance(raw, list) else []
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ids = sorted(
        {str(key) for point in timeline if isinstance(point, dict) for key in point.get(field, {})}
    )
    for item_id in ids:
        axis.step(
            [float(point["time"]) for point in timeline],
            [float(point.get(field, {}).get(item_id, 0.0)) for point in timeline],
            where="post",
            label=item_id,
        )
    _finish(axis, ylabel, ylabel, ids)
    _save(figure, path)


def _plot_fleet(metrics: dict[str, Any], path: Path) -> None:
    raw = metrics.get("fleet_capacity_timeline", [])
    timeline = raw if isinstance(raw, list) else []
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for field in ("active_gpus", "schedulable_gpus", "revocable_gpus"):
        axis.step(
            [float(point["time"]) for point in timeline],
            [float(point.get(field, 0.0)) for point in timeline],
            where="post",
            label=field,
        )
    _finish(axis, "Fleet capacity", "GPU count", ["capacity"] if timeline else [])
    _save(figure, path)


def _queue_ids(timeline: Any) -> list[str]:
    if not isinstance(timeline, list):
        return []
    return sorted(
        {
            str(queue_id)
            for point in timeline
            if isinstance(point, dict)
            for queue_id in point.get("queues", {})
            if queue_id != "root"
        }
    )


def _finish(axis: Any, title: str, ylabel: str, labels: list[str]) -> None:
    axis.set_title(title)
    axis.set_xlabel("Logical time")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if labels:
        axis.legend(fontsize=8)


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
