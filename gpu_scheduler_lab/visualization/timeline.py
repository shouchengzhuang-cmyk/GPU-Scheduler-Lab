from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from gpu_scheduler_lab.models.events import EventType, TraceRecord


def _segments(trace: list[TraceRecord]) -> list[tuple[str, str, float, float]]:
    active: dict[tuple[str, str], float] = {}
    segments: list[tuple[str, str, float, float]] = []
    for record in trace:
        if record.event in {EventType.JOB_START, EventType.JOB_RESUME, EventType.JOB_RESTART}:
            for gpu_id in record.gpu_ids:
                active.setdefault((record.job_id, gpu_id), record.time)
        elif record.event in {
            EventType.JOB_COMPLETE,
            EventType.JOB_PREEMPT,
            EventType.JOB_CHECKPOINT_COMPLETE,
        }:
            for gpu_id in record.gpu_ids:
                started = active.pop((record.job_id, gpu_id), None)
                if started is not None:
                    segments.append((gpu_id, record.job_id, started, record.time - started))
    return segments


def plot_timeline(trace: list[TraceRecord], path: Path, *, max_gpus: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    segments = _segments(trace)
    gpu_ids = sorted({gpu_id for gpu_id, _, _, _ in segments})[:max_gpus]
    row = {gpu_id: index for index, gpu_id in enumerate(gpu_ids)}
    figure_height = max(3.5, len(gpu_ids) * 0.38 + 1.5)
    figure, axis = plt.subplots(figsize=(12, figure_height), constrained_layout=True)
    labels = sorted({job_id for gpu_id, job_id, _, _ in segments if gpu_id in row})
    colors = {label: f"C{index % 10}" for index, label in enumerate(labels)}
    for gpu_id, job_id, start, duration in segments:
        if gpu_id not in row:
            continue
        axis.broken_barh(
            [(start, duration)],
            (row[gpu_id] - 0.38, 0.76),
            facecolors=colors[job_id],
            label=job_id,
        )
        axis.text(start + duration / 2, row[gpu_id], job_id, ha="center", va="center", fontsize=7)
    axis.set_yticks(range(len(gpu_ids)), gpu_ids)
    axis.set_xlabel("Logical time")
    axis.set_title("GPU allocation timeline")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(path, dpi=150)
    plt.close(figure)
