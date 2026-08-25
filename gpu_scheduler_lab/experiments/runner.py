from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_scheduler_lab.experiments.aggregation import aggregate_runs
from gpu_scheduler_lab.experiments.config import ExperimentConfig
from gpu_scheduler_lab.experiments.manifest import git_sha, python_version, scenario_hash
from gpu_scheduler_lab.scenario import Scenario, load_scenario
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.simulator.engine import Simulator
from gpu_scheduler_lab.visualization.experiment import plot_experiment_summary
from gpu_scheduler_lab.workload import GeneratorConfig, generate_scenario


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    manifest: Path
    runs: Path
    summary_csv: Path
    summary_json: Path
    comparison: Path


def run_experiment(config_path: Path) -> ExperimentArtifacts:
    config = ExperimentConfig.load(config_path)
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    revision = git_sha(config_path.parent)
    runs: list[dict[str, Any]] = []
    for seed in config.seeds:
        scenario = _scenario_for_seed(config, seed)
        fingerprint = scenario_hash(scenario)
        for scheduler_name in config.schedulers:
            result = Simulator(
                scenario.cluster,
                scenario.jobs,
                create_scheduler(scheduler_name),
            ).run()
            runs.append(
                {
                    "scheduler": scheduler_name,
                    "seed": seed,
                    "scenario_hash": fingerprint,
                    "metrics": result.metrics,
                    "result": result.to_dict(include_trace=True),
                }
            )
    summary = aggregate_runs(runs)
    manifest_runs = [
        {
            "scheduler": run["scheduler"],
            "seed": run["seed"],
            "scenario_hash": run["scenario_hash"],
            "metrics": run["metrics"],
        }
        for run in runs
    ]
    identity = hashlib.sha256(
        json.dumps(
            {
                "name": config.name,
                "timestamp": timestamp,
                "schedulers": config.schedulers,
                "seeds": config.seeds,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    trace_metadata = _trace_metadata(config, runs)
    scenario_hashes = sorted({str(run["scenario_hash"]) for run in runs})
    manifest = {
        "experiment_id": identity,
        "name": config.name,
        "timestamp": timestamp,
        "git_sha": revision,
        "python_version": python_version(),
        "scenario_hash": scenario_hashes[0] if len(scenario_hashes) == 1 else None,
        "scenario_hashes": scenario_hashes,
        "config": {
            "workload": config.workload,
            "schedulers": list(config.schedulers),
            "seeds": list(config.seeds),
        },
        **trace_metadata,
        "runs": manifest_runs,
    }
    artifacts = ExperimentArtifacts(
        manifest=output / "manifest.json",
        runs=output / "runs.json",
        summary_csv=output / "summary.csv",
        summary_json=output / "summary.json",
        comparison=output / "comparison.png",
    )
    _write_json(artifacts.manifest, manifest)
    _write_json(artifacts.runs, {"runs": runs})
    _write_json(artifacts.summary_json, {"summary": summary})
    _write_summary_csv(artifacts.summary_csv, summary)
    plot_experiment_summary(summary, artifacts.comparison)
    return artifacts


def _scenario_for_seed(config: ExperimentConfig, seed: int) -> Scenario:
    workload_type = str(config.workload.get("type", "scenario"))
    if workload_type in {"scenario", "trace"}:
        raw_path = config.workload.get("scenario")
        if raw_path is None:
            raise ValueError("trace/scenario workload requires workload.scenario")
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        return load_scenario(path)
    raw_generator = config.workload.get("generator", {})
    if not isinstance(raw_generator, dict):
        raise ValueError("workload.generator must be a mapping")
    values = dict(raw_generator)
    values["seed"] = seed
    return generate_scenario(GeneratorConfig(**values))


def _trace_metadata(config: ExperimentConfig, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if str(config.workload.get("type")) != "trace":
        return {"trace_source": None, "trace_version": None}
    scenario = _scenario_for_seed(config, config.seeds[0])
    return {
        "trace_source": scenario.metadata.get("source_url"),
        "trace_version": scenario.metadata.get("trace_version"),
        "scenario_hash": runs[0]["scenario_hash"] if runs else None,
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_summary_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fields = (
        list(rows[0])
        if rows
        else ["scheduler", "metric", "runs", "mean", "stddev", "median", "p95"]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
