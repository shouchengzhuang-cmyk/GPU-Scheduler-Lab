from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    workload: dict[str, Any]
    schedulers: tuple[str, ...]
    seeds: tuple[int, ...]
    output_directory: Path
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("experiment config root must be a mapping")
        experiment = raw.get("experiment", {})
        workload = raw.get("workload", {})
        output = raw.get("output", {})
        if not isinstance(experiment, dict) or not isinstance(workload, dict):
            raise ValueError("experiment and workload must be mappings")
        if not isinstance(output, dict):
            raise ValueError("output must be a mapping")
        name = str(experiment.get("name", "")).strip()
        if not name:
            raise ValueError("experiment.name must not be empty")
        workload_type = str(workload.get("type", "scenario"))
        if workload_type not in {"scenario", "trace", "synthetic"}:
            raise ValueError("workload.type must be scenario, trace, or synthetic")
        schedulers_raw = raw.get("schedulers", [])
        if not isinstance(schedulers_raw, list) or not schedulers_raw:
            raise ValueError("schedulers must be a non-empty list")
        schedulers = tuple(str(value) for value in schedulers_raw)
        seeds_raw = raw.get("seeds", [1])
        if not isinstance(seeds_raw, list) or not seeds_raw:
            raise ValueError("seeds must be a non-empty list")
        seeds = tuple(int(value) for value in seeds_raw)
        directory = Path(str(output.get("directory", f"experiment-results/{name}")))
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        return cls(
            name=name,
            workload=dict(workload),
            schedulers=schedulers,
            seeds=seeds,
            output_directory=directory,
            source_path=path,
        )
