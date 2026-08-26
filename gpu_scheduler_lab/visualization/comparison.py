from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from gpu_scheduler_lab.simulator.engine import SimulationResult


def plot_comparison(results: list[SimulationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [result.scheduler for result in results]
    definitions = (
        ("average_gpu_utilization", "Average GPU utilization"),
        ("average_waiting_time", "Average waiting time"),
        ("gpu_fragmentation_ratio", "Fragmentation ratio"),
        ("sla_violation_rate", "SLA violation rate"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, definitions, strict=True):
        values = [float(result.metrics[key]) for result in results]
        axis.bar(names, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("GPU scheduler comparison")
    figure.savefig(path, dpi=150)
    plt.close(figure)
