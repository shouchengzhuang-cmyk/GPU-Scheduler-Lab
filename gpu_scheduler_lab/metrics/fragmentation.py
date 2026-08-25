from __future__ import annotations

from gpu_scheduler_lab.models.cluster import Cluster


def gpu_count_fragmentation(cluster: Cluster) -> float:
    """Weighted node partiality; 0 means every node is full or empty, 1 means half free."""
    total = cluster.total_gpu_count
    if total == 0:
        return 0.0
    weighted_partiality = 0.0
    for node in cluster.nodes:
        capacity = len(node.gpus)
        if capacity == 0:
            continue
        free_fraction = node.free_gpu_count / capacity
        weighted_partiality += capacity * 4.0 * free_fraction * (1.0 - free_fraction)
    return weighted_partiality / total


def gpu_memory_fragmentation(cluster: Cluster) -> float:
    """Fraction of memory on occupied exclusive GPUs stranded by sub-capacity requests."""
    occupied_capacity = sum(gpu.memory_capacity_gb for gpu in cluster.gpus if gpu.occupied)
    if occupied_capacity == 0:
        return 0.0
    stranded = sum(
        gpu.memory_capacity_gb - gpu.allocated_memory_gb for gpu in cluster.gpus if gpu.occupied
    )
    return stranded / occupied_capacity


def fragmentation_snapshot(cluster: Cluster) -> tuple[float, float, float]:
    count = gpu_count_fragmentation(cluster)
    memory = gpu_memory_fragmentation(cluster)
    return count, memory, (count + memory) / 2.0
