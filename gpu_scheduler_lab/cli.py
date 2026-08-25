from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gpu_scheduler_lab.integrations.mini_ai_cloud import import_mini_ai_cloud_export
from gpu_scheduler_lab.scenario import Scenario, load_scenario, write_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.simulator.engine import SimulationResult, Simulator
from gpu_scheduler_lab.visualization import plot_comparison, plot_timeline
from gpu_scheduler_lab.workload import GeneratorConfig, generate_scenario

SCHEDULERS = ("fifo", "binpack", "spread", "preemptive")


def _weighted_ints(value: str) -> tuple[tuple[int, float], ...]:
    try:
        return tuple(
            (int(item.split(":", 1)[0]), float(item.split(":", 1)[1])) for item in value.split(",")
        )
    except (ValueError, IndexError) as exc:
        raise argparse.ArgumentTypeError("use VALUE:WEIGHT pairs separated by commas") from exc


def _weighted_floats(value: str) -> tuple[tuple[float, float], ...]:
    try:
        return tuple(
            (float(item.split(":", 1)[0]), float(item.split(":", 1)[1]))
            for item in value.split(",")
        )
    except (ValueError, IndexError) as exc:
        raise argparse.ArgumentTypeError("use VALUE:WEIGHT pairs separated by commas") from exc


def _priority_weights(value: str) -> tuple[float, float, float, float]:
    try:
        weights = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("priority weights must be numeric") from exc
    if len(weights) != 4:
        raise argparse.ArgumentTypeError("provide low,normal,high,critical weights")
    return weights


def _run(scenario: Scenario, scheduler_name: str) -> SimulationResult:
    return Simulator(scenario.cluster, scenario.jobs, create_scheduler(scheduler_name)).run()


def _scalar_metrics(result: SimulationResult) -> dict[str, str | int | float]:
    return {
        key: value
        for key, value in result.metrics.items()
        if isinstance(value, str | int | float) and not isinstance(value, bool)
    }


def _write_outputs(
    results: list[SimulationResult], output_dir: Path, stem: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    payload = {
        "results": [result.to_dict(include_trace=True) for result in results],
    }
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    rows = [{"scheduler": result.scheduler, **_scalar_metrics(result)} for result in results]
    fieldnames = list(rows[0]) if rows else ["scheduler"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _summary(results: list[SimulationResult]) -> str:
    header = (
        f"{'Scheduler':<12} {'Util':>8} {'Avg Wait':>10} {'P95 Wait':>10} "
        f"{'Fragment':>10} {'SLA Viol':>9}"
    )
    rows = [header, "-" * len(header)]
    for result in results:
        metrics = result.metrics
        rows.append(
            f"{result.scheduler:<12} {metrics['average_gpu_utilization']:>8.3f} "
            f"{metrics['average_waiting_time']:>10.2f} "
            f"{metrics['p95_waiting_time']:>10.2f} "
            f"{metrics['gpu_fragmentation_ratio']:>10.3f} "
            f"{metrics['sla_violation_rate']:>9.3f}"
        )
    return "\n".join(rows)


def _benchmark(args: argparse.Namespace) -> None:
    scenario = load_scenario(args.scenario)
    result = _run(scenario, args.scheduler)
    json_path, csv_path = _write_outputs(
        [result], args.output_dir, f"{args.scenario.stem}-{args.scheduler}"
    )
    timeline_path = args.output_dir / f"{args.scenario.stem}-{args.scheduler}-timeline.png"
    if not args.no_charts:
        plot_timeline(result.trace, timeline_path)
    print(_summary([result]))
    print(f"JSON: {json_path}\nCSV: {csv_path}")
    if not args.no_charts:
        print(f"Timeline: {timeline_path}")


def _compare(args: argparse.Namespace) -> None:
    scenario = load_scenario(args.scenario)
    names = [name.strip() for name in args.schedulers.split(",") if name.strip()]
    results = [_run(scenario, name) for name in names]
    json_path, csv_path = _write_outputs(results, args.output_dir, f"{args.scenario.stem}-compare")
    chart_path = args.output_dir / f"{args.scenario.stem}-comparison.png"
    timeline_path = args.output_dir / f"{args.scenario.stem}-{results[-1].scheduler}-timeline.png"
    if not args.no_charts:
        plot_comparison(results, chart_path)
        plot_timeline(results[-1].trace, timeline_path)
    print(_summary(results))
    print(f"JSON: {json_path}\nCSV: {csv_path}")
    if not args.no_charts:
        print(f"Comparison chart: {chart_path}\nTimeline: {timeline_path}")


def _generate(args: argparse.Namespace) -> None:
    config = GeneratorConfig(
        job_count=args.jobs,
        node_count=args.nodes,
        gpus_per_node=args.gpus_per_node,
        arrival_rate=args.arrival_rate,
        median_duration=args.median_duration,
        duration_distribution=args.duration_distribution,
        gpu_count_distribution=args.gpu_count_distribution,
        gpu_memory_distribution=args.gpu_memory_distribution,
        priority_weights=args.priority_weights,
        training_ratio=args.training_ratio,
        gang_probability=args.gang_probability,
        sla_probability=args.sla_probability,
        seed=args.seed,
        profile=args.profile,
    )
    write_scenario(generate_scenario(config), args.output)
    print(f"Scenario: {args.output}")


def _import_mini_ai_cloud(args: argparse.Namespace) -> None:
    with args.input.open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    scenario = import_mini_ai_cloud_export(payload)
    write_scenario(scenario, args.output)
    print(
        f"Scenario: {args.output} ({len(scenario.cluster.nodes)} nodes, {len(scenario.jobs)} jobs)"
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-charts", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic GPU scheduling laboratory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="run one scheduler")
    benchmark.add_argument("--scenario", type=Path, required=True)
    benchmark.add_argument("--scheduler", choices=SCHEDULERS, required=True)
    _add_output_args(benchmark)
    benchmark.set_defaults(handler=_benchmark)

    compare = subparsers.add_parser("compare", help="compare schedulers on one workload")
    compare.add_argument("--scenario", type=Path, required=True)
    compare.add_argument("--schedulers", default=",".join(SCHEDULERS))
    _add_output_args(compare)
    compare.set_defaults(handler=_compare)

    generate = subparsers.add_parser("generate", help="write a seeded synthetic scenario")
    generate.add_argument("--profile", choices=("mixed", "fragmentation", "burst"), required=True)
    generate.add_argument("--jobs", type=int, default=100)
    generate.add_argument("--nodes", type=int, default=8)
    generate.add_argument("--gpus-per-node", type=int, default=8)
    generate.add_argument("--arrival-rate", type=float, default=1.0)
    generate.add_argument("--median-duration", type=float, default=60.0)
    generate.add_argument(
        "--duration-distribution",
        choices=("fixed", "exponential", "lognormal"),
        default="lognormal",
    )
    generate.add_argument(
        "--gpu-count-distribution",
        type=_weighted_ints,
        help="weighted values, for example 1:0.6,2:0.3,4:0.1",
    )
    generate.add_argument(
        "--gpu-memory-distribution",
        type=_weighted_floats,
        help="weighted GiB values, for example 20:0.7,40:0.3",
    )
    generate.add_argument(
        "--priority-weights",
        type=_priority_weights,
        default=None,
        help="low,normal,high,critical weights",
    )
    generate.add_argument("--training-ratio", type=float, default=0.35)
    generate.add_argument("--gang-probability", type=float, default=0.35)
    generate.add_argument("--sla-probability", type=float, default=0.5)
    generate.add_argument("--seed", type=int, default=20260825)
    generate.add_argument("--output", type=Path, required=True)
    generate.set_defaults(handler=_generate)

    importer = subparsers.add_parser(
        "import-mini-ai-cloud", help="convert a Mini-AI-Cloud v1 file export"
    )
    importer.add_argument("--input", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.set_defaults(handler=_import_mini_ai_cloud)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
