from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from gpu_scheduler_lab.models.cluster import GPU, Cluster, Node
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.models.topology import (
    TopologyMode,
    topology_distance,
    topology_domain,
    topology_requirement_satisfied,
)
from gpu_scheduler_lab.schedulers.base import Scheduler


class TopologyAwareScheduler(Scheduler):
    """Choose a deterministic feasible placement with hierarchical locality."""

    name = "topology"

    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        eligible = sorted(
            cluster.eligible_gpus(job),
            key=lambda gpu: (gpu.memory_capacity_gb - job.gpu_memory_gb, gpu.node_id, gpu.id),
        )
        if len(eligible) < job.requested_gpu_count:
            return None
        nodes = {node.id: node for node in cluster.schedulable_nodes}
        candidates = self._candidate_placements(eligible, nodes, job.requested_gpu_count)
        topologies = {node_id: node.topology for node_id, node in nodes.items()}
        feasible = [
            placement
            for placement in candidates
            if topology_requirement_satisfied(
                job.topology_mode,
                (gpu.node_id for gpu in placement),
                topologies,
            )
        ]
        if not feasible:
            return None
        selected = min(feasible, key=lambda item: self._score(cluster, nodes, item, job))
        return sorted(gpu.id for gpu in selected)

    @staticmethod
    def _candidate_placements(
        eligible: list[GPU], nodes: dict[str, Node], required: int
    ) -> list[tuple[GPU, ...]]:
        candidates: dict[tuple[str, ...], tuple[GPU, ...]] = {}

        def add(items: Iterable[GPU]) -> None:
            placement = tuple(list(items)[:required])
            if len(placement) == required:
                candidates[tuple(sorted(gpu.id for gpu in placement))] = placement

        add(eligible)
        for level in ("node", "rack", "zone"):
            grouped: dict[str, list[GPU]] = defaultdict(list)
            for gpu in eligible:
                node = nodes[gpu.node_id]
                grouped[topology_domain(node.id, node.topology, level)].append(gpu)
            for domain in sorted(grouped):
                add(grouped[domain])

        for seed_id in sorted({gpu.node_id for gpu in eligible}):
            seed = nodes[seed_id]
            ordered_nodes = sorted(
                nodes.values(),
                key=lambda node: (
                    topology_distance(seed.id, seed.topology, node.id, node.topology),
                    node.id,
                ),
            )
            by_node = {
                node.id: [gpu for gpu in eligible if gpu.node_id == node.id]
                for node in nodes.values()
            }
            add(gpu for node in ordered_nodes for gpu in by_node[node.id])
        return list(candidates.values())

    @staticmethod
    def _score(
        cluster: Cluster, nodes: dict[str, Node], placement: tuple[GPU, ...], job: Job
    ) -> tuple[float | int | tuple[str, ...], ...]:
        node_ids = sorted({gpu.node_id for gpu in placement})
        racks = {topology_domain(node_id, nodes[node_id].topology, "rack") for node_id in node_ids}
        if job.topology_mode in {
            TopologyMode.PREFER_SAME_RACK,
            TopologyMode.REQUIRE_SAME_RACK,
        }:
            primary_domains = len(racks)
        else:
            primary_domains = len(node_ids)
        distances = [
            topology_distance(
                left,
                nodes[left].topology,
                right,
                nodes[right].topology,
            )
            for index, left in enumerate(node_ids)
            for right in node_ids[index + 1 :]
        ]
        max_distance = max(distances, default=0)
        average_distance = sum(distances) / len(distances) if distances else 0.0
        fragmentation_delta = _count_fragmentation_delta(cluster, placement)
        memory_waste = sum(gpu.memory_capacity_gb - job.gpu_memory_gb for gpu in placement)
        return (
            primary_domains,
            max_distance,
            average_distance,
            fragmentation_delta,
            memory_waste,
            tuple(sorted(gpu.id for gpu in placement)),
        )


def _count_fragmentation_delta(cluster: Cluster, placement: tuple[GPU, ...]) -> float:
    total = cluster.total_gpu_count
    if total == 0:
        return 0.0
    additions: dict[str, int] = defaultdict(int)
    for gpu in placement:
        additions[gpu.node_id] += 1

    def partiality(capacity: int, occupied: int) -> float:
        free_fraction = (capacity - occupied) / capacity
        return capacity * 4.0 * free_fraction * (1.0 - free_fraction)

    delta = 0.0
    for node in cluster.schedulable_nodes:
        if not node.gpus:
            continue
        before = partiality(len(node.gpus), node.occupied_gpu_count)
        after = partiality(len(node.gpus), node.occupied_gpu_count + additions.get(node.id, 0))
        delta += after - before
    return delta / total
