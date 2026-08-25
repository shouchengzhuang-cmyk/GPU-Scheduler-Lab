from gpu_scheduler_lab.models.cluster import GPU, Cluster, Node
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.models.topology import TopologyMode
from gpu_scheduler_lab.schedulers.base import Scheduler


class SpreadScheduler(Scheduler):
    """Use least-loaded nodes and round-robin across fault domains."""

    name = "spread"

    @staticmethod
    def _eligible(node: Node, job: Job) -> list[GPU]:
        return sorted(
            (gpu for gpu in node.gpus if gpu.can_host(job)),
            key=lambda gpu: (gpu.memory_capacity_gb - job.gpu_memory_gb, gpu.id),
        )

    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        if job.topology_mode in {
            TopologyMode.REQUIRE_SAME_NODE,
            TopologyMode.REQUIRE_SAME_RACK,
        }:
            from gpu_scheduler_lab.schedulers.topology import TopologyAwareScheduler

            return TopologyAwareScheduler().place(cluster, job)
        nodes = [node for node in cluster.nodes if node.schedulable]
        nodes.sort(key=lambda node: (node.occupied_gpu_count, node.id))
        available = {node.id: self._eligible(node, job) for node in nodes}
        if sum(len(gpus) for gpus in available.values()) < job.gpu_count:
            return None

        placement: list[str] = []
        index = 0
        while len(placement) < job.gpu_count:
            made_progress = False
            for node in nodes:
                gpus = available[node.id]
                if index < len(gpus):
                    placement.append(gpus[index].id)
                    made_progress = True
                    if len(placement) == job.gpu_count:
                        return placement
            if not made_progress:
                break
            index += 1
        return None
