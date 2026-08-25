from gpu_scheduler_lab.models.cluster import GPU, Cluster, Node
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.models.topology import TopologyMode
from gpu_scheduler_lab.schedulers.base import Scheduler


class BinPackScheduler(Scheduler):
    """Pack onto busy nodes, then choose tight-memory GPUs deterministically."""

    name = "binpack"

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
        candidates = [
            (node, self._eligible(node, job)) for node in cluster.nodes if node.schedulable
        ]
        if sum(len(gpus) for _, gpus in candidates) < job.gpu_count:
            return None

        # Occupied nodes come first. Among equally loaded nodes, consuming the smaller
        # free block first preserves larger intact blocks for future gang jobs.
        candidates.sort(
            key=lambda item: (
                item[0].occupied_gpu_count == 0,
                -item[0].occupied_gpu_count,
                len(item[1]),
                sum(gpu.memory_capacity_gb - job.gpu_memory_gb for gpu in item[1]),
                item[0].id,
            )
        )
        placement: list[str] = []
        for _, gpus in candidates:
            needed = job.gpu_count - len(placement)
            placement.extend(gpu.id for gpu in gpus[:needed])
            if len(placement) == job.gpu_count:
                return placement
        return None
