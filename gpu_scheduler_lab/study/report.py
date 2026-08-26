from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from gpu_scheduler_lab.study.artifacts import sha256_file

SummaryValue = str | int | float
SummaryRow = dict[str, SummaryValue]


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    output_directory: Path
    report: Path
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    hashes: Path


def generate_study_report(input_directory: Path) -> ReportArtifacts:
    root = input_directory.resolve()
    manifest = _load_mapping(root / "manifest.json", "study manifest")
    summary = _load_summary(root / "summary.json")
    _load_mapping(root / "environment.json", "study environment")
    _load_mapping(root / "scenario-hashes.json", "scenario hashes")
    _load_runs(root / "runs.json")

    tables_dir = root / "tables"
    figures_dir = root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    baseline = [row for row in summary if str(row["variant_id"]) == "baseline"]
    sensitivity = [row for row in summary if str(row["variant_id"]).startswith("sensitivity-")]
    ablation = [row for row in summary if str(row["variant_id"]).startswith("ablation-")]
    table_paths = (
        tables_dir / "baseline.csv",
        tables_dir / "sensitivity.csv",
        tables_dir / "ablation.csv",
    )
    for path, rows in zip(table_paths, (baseline, sensitivity, ablation), strict=True):
        _write_summary_csv(path, rows)

    figure_paths = (
        figures_dir / "baseline-key-metrics.png",
        figures_dir / "sensitivity-completion-rate.png",
        figures_dir / "ablation-completion-rate.png",
    )
    _plot_baseline(baseline, figure_paths[0])
    _plot_variant_metric(
        sensitivity,
        figure_paths[1],
        title="Sensitivity: completion rate",
    )
    _plot_variant_metric(
        ablation,
        figure_paths[2],
        title="Ablation: completion rate",
    )

    report_path = root / "report.md"
    report_path.write_text(
        _render_report(manifest, baseline, sensitivity, ablation),
        encoding="utf-8",
        newline="\n",
    )
    hashes_path = root / "hashes.sha256"
    write_hash_manifest(root, hashes_path)
    verify_hash_manifest(root, hashes_path)
    return ReportArtifacts(root, report_path, table_paths, figure_paths, hashes_path)


def write_hash_manifest(root: Path, destination: Path) -> None:
    root = root.resolve()
    destination = destination.resolve()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.resolve() != destination),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    lines: list[str] = []
    for path in files:
        if path.is_symlink():
            raise ValueError(f"study artifact must not be a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def verify_hash_manifest(root: Path, manifest_path: Path | None = None) -> int:
    root = root.resolve()
    hashes = (manifest_path or root / "hashes.sha256").resolve()
    if root not in hashes.parents or not hashes.is_file() or hashes.is_symlink():
        raise ValueError("hash manifest is missing or outside the study bundle")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(hashes.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not relative
            or relative in entries
        ):
            raise ValueError(f"invalid hash manifest line {line_number}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError(f"unsafe artifact path on hash manifest line {line_number}")
        entries[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != hashes
    }
    if actual != set(entries):
        missing = sorted(actual - set(entries))
        extra = sorted(set(entries) - actual)
        raise ValueError(f"hash manifest coverage mismatch: missing={missing}, extra={extra}")
    for relative, expected in sorted(entries.items()):
        path = (root / relative).resolve()
        if root not in path.parents or path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")
    return len(entries)


def _load_json(path: Path, label: str) -> object:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label}: {path}") from exc
    return payload


def _load_mapping(path: Path, label: str) -> dict[str, object]:
    payload = _load_json(path, label)
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{label} must be a JSON object")
    return {str(key): value for key, value in payload.items()}


def _load_summary(path: Path) -> list[SummaryRow]:
    payload = _load_mapping(path, "study summary")
    raw_rows = payload.get("summary")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("study summary must contain non-empty rows")
    rows: list[SummaryRow] = []
    required = ("policy_id", "variant_id", "metric_id", "samples", "mean", "stddev")
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"study summary row {index} must be an object")
        if any(key not in raw for key in required):
            raise ValueError(f"study summary row {index} is incomplete")
        policy_id = raw["policy_id"]
        variant_id = raw["variant_id"]
        metric_id = raw["metric_id"]
        samples = raw["samples"]
        mean_value = raw["mean"]
        stddev = raw["stddev"]
        if not all(
            isinstance(value, str) and value for value in (policy_id, variant_id, metric_id)
        ):
            raise ValueError(f"study summary row {index} has invalid identifiers")
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise ValueError(f"study summary row {index} has invalid sample count")
        if any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in (mean_value, stddev)
        ):
            raise ValueError(f"study summary row {index} has invalid statistics")
        rows.append(
            {
                "policy_id": policy_id,
                "variant_id": variant_id,
                "metric_id": metric_id,
                "samples": samples,
                "mean": float(mean_value),
                "stddev": float(stddev),
            }
        )
    return rows


def _load_runs(path: Path) -> None:
    payload = _load_mapping(path, "study runs")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs or any(not isinstance(run, dict) for run in runs):
        raise ValueError("study runs must contain non-empty run objects")


def _write_summary_csv(path: Path, rows: list[SummaryRow]) -> None:
    fields = ["policy_id", "variant_id", "metric_id", "samples", "mean", "stddev"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_baseline(rows: list[SummaryRow], path: Path) -> None:
    metrics = (
        "average-gpu-utilization",
        "p95-wait",
        "completion-rate",
        "jain-service-quality-fairness",
    )
    grouped: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["metric_id"])][str(row["policy_id"])] = (
            float(row["mean"]),
            float(row["stddev"]),
        )
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics, strict=True):
        policies = sorted(grouped[metric])
        values = [grouped[metric][policy][0] for policy in policies]
        errors = [grouped[metric][policy][1] for policy in policies]
        axis.bar(policies, values, yerr=errors, capsize=3)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Baseline key metrics (error bars: population standard deviation)")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_variant_metric(rows: list[SummaryRow], path: Path, *, title: str) -> None:
    selected = [row for row in rows if str(row["metric_id"]) == "completion-rate"]
    variants = sorted({str(row["variant_id"]) for row in selected})
    policies = sorted({str(row["policy_id"]) for row in selected})
    values = {
        (str(row["policy_id"]), str(row["variant_id"])): float(row["mean"]) for row in selected
    }
    figure, axis = plt.subplots(
        figsize=(max(9.0, len(variants) * 0.75), 5), constrained_layout=True
    )
    if variants and policies:
        positions = list(range(len(variants)))
        for policy in policies:
            axis.plot(
                positions,
                [values.get((policy, variant), math.nan) for variant in variants],
                marker="o",
                label=policy,
            )
        axis.legend()
        labels = [
            variant.removeprefix("sensitivity-").removeprefix("ablation-") for variant in variants
        ]
        axis.set_xticks(positions, labels, rotation=30, ha="right")
    else:
        axis.text(0.5, 0.5, "No matching rows", ha="center", va="center")
    axis.set_title(title)
    axis.set_ylabel("mean completion-rate")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _render_report(
    manifest: dict[str, object],
    baseline: list[SummaryRow],
    sensitivity: list[SummaryRow],
    ablation: list[SummaryRow],
) -> str:
    title = _required_text(manifest, "title")
    question = _required_text(manifest, "research_question")
    source_config = _required_text(manifest, "source_config")
    git_sha = _required_text(manifest, "git_sha")
    dirty_value = manifest.get("dirty_tree")
    dirty = "UNKNOWN" if dirty_value is None else ("DIRTY" if dirty_value is True else "CLEAN")
    scenario = manifest.get("scenario")
    scenario_map = scenario if isinstance(scenario, dict) else {}
    limitations = scenario_map.get("limitations")
    scenario_limitations = (
        [str(item) for item in limitations if isinstance(item, str)]
        if isinstance(limitations, list)
        else []
    )
    policy_lines = _policy_lines(manifest.get("policies"))
    metric_lines = _metric_lines(manifest.get("metrics"))
    baseline_table = _markdown_summary_table(baseline)
    sensitivity_table = _markdown_summary_table(sensitivity[:16])
    ablation_table = _markdown_summary_table(ablation[:16])
    limitation_lines = "\n".join(f"- {item}" for item in scenario_limitations)
    return f"""# {title}

## Research question

{question}

## Audit identity

- Git SHA: `{git_sha}`
- Worktree state at run time: **{dirty}**
- Source config: `{source_config}`
- Run count: {manifest.get("run_count")}
- Seeds: {manifest.get("seeds")}
- Replications per seed: {manifest.get("replications_per_seed")}

## Model assumptions

- This is a deterministic discrete-event simulation with logical time.
- GPU devices are modeled as exclusive resources; modeled topology, preemption, reclaim,
  elasticity, and fleet events are abstractions rather than measurements of physical hardware.
{limitation_lines}

## Method

The runner expands the frozen one-at-a-time sensitivity variables and named ablations, executes
every formal policy over the declared seeds, and aggregates finite scalar metrics using the mean
and population standard deviation. Independent run manifests retain the exact plan and seed.

## Policies

{policy_lines}

## Data source and normalization

The included study uses a seeded synthetic scenario generator. No Alibaba tenant identities or
production workload fields are inferred. Logical time, GPU model mix, tenant assignment, topology
constraints, and revocable capacity are normalized from the checked-in scenario/config contract.

## Metrics

{metric_lines}

## Results

All values below are rendered from `summary.json`; they are not copied into this report by hand.

### Baseline

{baseline_table}

![Baseline key metrics](figures/baseline-key-metrics.png)

## Sensitivity

The complete sensitivity table is `tables/sensitivity.csv`. The excerpt preserves stable summary
ordering and reports mean plus population standard deviation.

{sensitivity_table}

![Completion-rate sensitivity](figures/sensitivity-completion-rate.png)

## Ablation

The complete ablation table is `tables/ablation.csv`. An ablation disables only the named modeled
mechanism; it does not prove causal effects outside this simulator.

{ablation_table}

![Completion-rate ablation](figures/ablation-completion-rate.png)

## Limitations

- No real NVIDIA GPU, CUDA, Kubernetes, network, or storage system was exercised.
- Simulator wall time is not cluster scheduler throughput.
- Synthetic tenant assignment is not production tenant data.
- Population standard deviation over configured seeds is descriptive, not a significance test.
- The CI fixture validates report orchestration and must not be used for portfolio conclusions.

## Reproducibility instructions

```bash
python -m gpu_scheduler_lab study run --config study/{source_config}
python -m gpu_scheduler_lab study report --input <output-directory>
python -m gpu_scheduler_lab study verify --input <output-directory>
```

`manifest.json`, `environment.json`, `scenario-hashes.json`, `runs.json`, the summary files,
tables, figures, per-run records, and this report are covered by `hashes.sha256` (the hash manifest
itself is excluded because it cannot self-hash).
"""


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"study manifest is missing {key}")
    return value


def _policy_lines(value: object) -> str:
    if not isinstance(value, list):
        raise ValueError("study manifest policies must be a list")
    lines: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("study manifest policy must be an object")
        identifier = item.get("id")
        description = item.get("description")
        mechanisms = item.get("mechanisms")
        if not isinstance(identifier, str) or not isinstance(description, str):
            raise ValueError("study manifest policy is incomplete")
        mechanism_text = (
            ", ".join(str(entry) for entry in mechanisms) if isinstance(mechanisms, list) else ""
        )
        lines.append(f"- **{identifier}**: {description} Mechanisms: {mechanism_text}.")
    return "\n".join(lines)


def _metric_lines(value: object) -> str:
    if not isinstance(value, list):
        raise ValueError("study manifest metrics must be a list")
    lines: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("study manifest metric must be an object")
        identifier = item.get("id")
        unit = item.get("unit")
        direction = item.get("direction")
        if not all(isinstance(entry, str) for entry in (identifier, unit, direction)):
            raise ValueError("study manifest metric is incomplete")
        lines.append(f"- `{identifier}` ({unit}, {direction})")
    return "\n".join(lines)


def _markdown_summary_table(rows: list[SummaryRow]) -> str:
    selected_metrics = {
        "average-gpu-utilization",
        "p95-wait",
        "completion-rate",
        "jain-service-quality-fairness",
    }
    selected = [row for row in rows if str(row["metric_id"]) in selected_metrics]
    selected = rows[:16] if not selected else selected[:16]
    lines = [
        "| Policy | Variant | Metric | Samples | Mean | Stddev |",
        "|---|---|---|---:|---:|---:|",
    ]
    lines.extend(
        "| {policy} | {variant} | {metric} | {samples} | {mean} | {stddev} |".format(
            policy=row["policy_id"],
            variant=row["variant_id"],
            metric=row["metric_id"],
            samples=row["samples"],
            mean=_format_number(float(row["mean"])),
            stddev=_format_number(float(row["stddev"])),
        )
        for row in selected
    )
    return "\n".join(lines)


def _format_number(value: float) -> str:
    return f"{value:.6g}"
