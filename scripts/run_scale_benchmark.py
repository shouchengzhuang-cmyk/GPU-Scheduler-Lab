from __future__ import annotations

import json
import time

from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.simulator.engine import Simulator
from gpu_scheduler_lab.workload import GeneratorConfig, generate_scenario


def main() -> None:
    config = GeneratorConfig(
        profile="mixed",
        node_count=100,
        gpus_per_node=8,
        job_count=10_000,
        arrival_rate=1.5,
        median_duration=120.0,
        seed=20260825,
    )
    scenario = generate_scenario(config)
    started = time.perf_counter()
    results = []
    for name in ("binpack", "spread"):
        result = Simulator(scenario.cluster, scenario.jobs, create_scheduler(name)).run()
        results.append(
            {
                "scheduler": name,
                "elapsed_seconds": result.elapsed_seconds,
                "metrics": result.metrics,
            }
        )
    print(
        json.dumps(
            {
                "cluster": {"nodes": 100, "gpus": 800},
                "jobs": 10_000,
                "seed": config.seed,
                "wall_seconds": time.perf_counter() - started,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
