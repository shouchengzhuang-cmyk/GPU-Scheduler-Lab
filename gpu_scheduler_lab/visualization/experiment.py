from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_experiment_summary(rows: list[dict[str, str | int | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["metric"])][str(row["scheduler"])] = (
            float(row["mean"]),
            float(row["stddev"]),
        )
    metrics = list(grouped)[:8]
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics, strict=False):
        names = sorted(grouped[metric])
        means = [grouped[metric][name][0] for name in names]
        errors = [grouped[metric][name][1] for name in names]
        axis.bar(names, means, yerr=errors, capsize=3)
        axis.set_title(metric.replace("_", " "))
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    for axis in list(axes.flat)[len(metrics) :]:
        axis.set_visible(False)
    figure.suptitle("Experiment summary (error bars: population standard deviation)")
    figure.savefig(path, dpi=150)
    plt.close(figure)
