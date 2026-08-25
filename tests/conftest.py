from __future__ import annotations

from gpu_scheduler_lab.models import GPU, Cluster, Node


def make_cluster(capacities: list[list[float]]) -> Cluster:
    return Cluster(
        [
            Node(
                id=f"node-{node_index}",
                gpus=[
                    GPU(
                        id=f"gpu-{node_index}-{gpu_index}",
                        node_id=f"node-{node_index}",
                        memory_capacity_gb=memory,
                    )
                    for gpu_index, memory in enumerate(node_capacities)
                ],
            )
            for node_index, node_capacities in enumerate(capacities)
        ]
    )
