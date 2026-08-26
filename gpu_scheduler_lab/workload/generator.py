from __future__ import annotations

import math
import random
from dataclasses import dataclass

from gpu_scheduler_lab.models.cluster import GPU, Cluster, Node
from gpu_scheduler_lab.models.job import Job, JobType, Priority
from gpu_scheduler_lab.models.topology import TopologyMode
from gpu_scheduler_lab.scenario import Scenario


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    job_count: int = 100
    node_count: int = 8
    gpus_per_node: int = 8
    arrival_rate: float = 1.0
    median_duration: float = 60.0
    duration_sigma: float = 0.65
    duration_distribution: str = "lognormal"
    gpu_count_distribution: tuple[tuple[int, float], ...] | None = None
    gpu_memory_distribution: tuple[tuple[float, float], ...] | None = None
    priority_weights: tuple[float, float, float, float] | None = None
    training_ratio: float = 0.35
    gang_probability: float = 0.35
    sla_probability: float = 0.5
    seed: int = 20260825
    profile: str = "mixed"

    def __post_init__(self) -> None:
        if self.job_count < 0:
            raise ValueError("job_count must be non-negative")
        if self.node_count <= 0 or self.gpus_per_node <= 0:
            raise ValueError("node_count and gpus_per_node must be positive")
        if self.arrival_rate <= 0 or self.median_duration <= 0:
            raise ValueError("arrival_rate and median_duration must be positive")
        if self.duration_sigma < 0:
            raise ValueError("duration_sigma must be non-negative")
        if self.duration_distribution not in {"fixed", "exponential", "lognormal"}:
            raise ValueError("duration_distribution must be fixed, exponential, or lognormal")
        if self.gpu_count_distribution is not None:
            if not self.gpu_count_distribution:
                raise ValueError("GPU count distribution must not be empty")
            if any(count <= 0 or weight <= 0 for count, weight in self.gpu_count_distribution):
                raise ValueError("GPU count distribution values and weights must be positive")
            cluster_gpu_count = self.node_count * self.gpus_per_node
            if any(count > cluster_gpu_count for count, _ in self.gpu_count_distribution):
                raise ValueError(
                    "GPU count distribution cannot request more GPUs than the cluster contains"
                )
        if self.gpu_memory_distribution is not None and any(
            memory <= 0 or weight <= 0 for memory, weight in self.gpu_memory_distribution
        ):
            raise ValueError("GPU memory distribution values and weights must be positive")
        if self.priority_weights is not None:
            if len(self.priority_weights) != 4 or any(
                weight < 0 for weight in self.priority_weights
            ):
                raise ValueError("priority_weights must contain four non-negative weights")
            if sum(self.priority_weights) <= 0:
                raise ValueError("priority_weights must contain at least one positive weight")
        for name, value in (
            ("training_ratio", self.training_ratio),
            ("gang_probability", self.gang_probability),
            ("sla_probability", self.sla_probability),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.profile not in {"mixed", "fragmentation", "burst", "topology", "backfill"}:
            raise ValueError("profile must be mixed, fragmentation, burst, topology, or backfill")

    @property
    def resolved_priority_weights(self) -> tuple[float, float, float, float]:
        if self.priority_weights is not None:
            return self.priority_weights
        if self.profile == "burst":
            return (15, 40, 35, 10)
        return (20, 50, 25, 5)


def _cluster(config: GeneratorConfig) -> Cluster:
    capacities = (24.0, 40.0, 80.0)
    models = ("A10", "A100-40GB", "A100-80GB")
    nodes = []
    for node_index in range(config.node_count):
        node_id = f"node-{node_index:03d}"
        memory = capacities[node_index % len(capacities)]
        nodes.append(
            Node(
                id=node_id,
                topology={
                    "zone": f"zone-{node_index // 20:02d}",
                    "rack": f"rack-{node_index // 5:02d}",
                },
                gpus=[
                    GPU(
                        id=f"{node_id}-gpu-{gpu_index}",
                        node_id=node_id,
                        memory_capacity_gb=memory,
                        model=models[node_index % len(models)],
                    )
                    for gpu_index in range(config.gpus_per_node)
                ],
            )
        )
    return Cluster(nodes)


def _arrival(rng: random.Random, config: GeneratorConfig, index: int, previous: float) -> float:
    if index == 0:
        return 0.0
    if config.profile == "burst":
        burst = index // 100
        return burst * 20.0 + rng.random() * 1.5
    if config.profile == "backfill":
        return float(index // 12) * 10.0 + rng.random()
    return previous + rng.expovariate(config.arrival_rate)


def generate_scenario(config: GeneratorConfig) -> Scenario:
    rng = random.Random(config.seed)
    jobs: list[Job] = []
    arrival = 0.0
    for index in range(config.job_count):
        arrival = _arrival(rng, config, index, arrival)
        training = rng.random() < config.training_ratio
        if config.gpu_memory_distribution is not None:
            memory_values, memory_weights = zip(*config.gpu_memory_distribution, strict=True)
            memory = rng.choices(memory_values, memory_weights)[0]
        elif config.profile in {"fragmentation", "topology"}:
            memory = rng.choices((10.0, 20.0, 23.0, 38.0, 70.0), (10, 25, 25, 20, 20))[0]
        elif config.profile == "burst":
            memory = rng.choices((8.0, 12.0, 20.0), (30, 45, 25))[0]
        else:
            memory = rng.choices((8.0, 16.0, 20.0, 38.0, 70.0), (15, 25, 25, 20, 15))[0]

        if config.gpu_count_distribution is not None:
            count_values, count_weights = zip(*config.gpu_count_distribution, strict=True)
            gpu_count = rng.choices(count_values, count_weights)[0]
        elif config.profile == "fragmentation":
            gpu_count = rng.choices((1, 2, 4, 8), (45, 25, 20, 10))[0]
        elif config.profile == "topology":
            gpu_count = rng.choices((2, 4, 8, 16), (20, 35, 35, 10))[0]
        elif config.profile == "backfill":
            gpu_count = min(
                config.node_count * config.gpus_per_node,
                config.gpus_per_node * 2 if index % 20 == 0 else rng.choice((1, 1, 2)),
            )
        elif config.profile == "burst":
            gpu_count = rng.choices((1, 2, 4), (78, 18, 4))[0]
        else:
            gpu_count = rng.choices((1, 2, 4, 8), (55, 25, 15, 5))[0]
        if config.gpu_count_distribution is None:
            gpu_count = min(gpu_count, config.node_count * config.gpus_per_node)
            if training:
                gpu_count = max(gpu_count, min(2, config.gpus_per_node))
        if config.duration_distribution == "fixed":
            sampled_duration = config.median_duration
        elif config.duration_distribution == "exponential":
            sampled_duration = rng.expovariate(1.0 / config.median_duration)
        else:
            sampled_duration = rng.lognormvariate(
                math.log(config.median_duration), config.duration_sigma
            )
        if config.profile == "backfill":
            duration = (
                config.median_duration * 4
                if index % 20 == 0
                else max(1.0, config.median_duration * 0.15)
            )
        else:
            duration = max(1.0, sampled_duration * (2.0 if training else 0.45))
        priority = rng.choices(
            (Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.CRITICAL),
            config.resolved_priority_weights,
        )[0]
        sla_deadline = None
        if rng.random() < config.sla_probability:
            sla_deadline = arrival + duration * rng.uniform(1.15, 2.5)
        topology_mode = TopologyMode.NONE
        gpu_model: str | None = None
        allowed_gpu_models: tuple[str, ...] = ()
        if config.profile == "topology":
            topology_mode = rng.choice(
                (
                    TopologyMode.PREFER_SAME_NODE,
                    TopologyMode.PREFER_SAME_RACK,
                    TopologyMode.REQUIRE_SAME_RACK,
                )
            )
            if memory > 40:
                gpu_model = "A100-80GB"
            elif memory > 24:
                allowed_gpu_models = ("A100-40GB", "A100-80GB")
        jobs.append(
            Job(
                id=f"{config.profile}-{index:05d}",
                arrival_time=arrival,
                duration=duration,
                gpu_count=gpu_count,
                gpu_memory_gb=memory,
                priority=priority,
                job_type=JobType.TRAINING if training else JobType.INFERENCE,
                gang=training or rng.random() < config.gang_probability,
                sla_deadline=sla_deadline,
                group="training" if training else "inference",
                gpu_model=gpu_model,
                allowed_gpu_models=allowed_gpu_models,
                topology_mode=topology_mode,
            )
        )
    return Scenario(
        cluster=_cluster(config),
        jobs=jobs,
        metadata={
            "profile": config.profile,
            "seed": config.seed,
            "generator": "gpu-scheduler-lab",
        },
    )
