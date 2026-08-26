from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.models.topology import TopologyMode
from gpu_scheduler_lab.schedulers.base import Scheduler


class FIFOScheduler(Scheduler):
    name = "fifo"

    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        if job.topology_mode in {
            TopologyMode.REQUIRE_SAME_NODE,
            TopologyMode.REQUIRE_SAME_RACK,
        }:
            from gpu_scheduler_lab.schedulers.topology import TopologyAwareScheduler

            return TopologyAwareScheduler().place(cluster, job)
        eligible = cluster.eligible_gpus(job)
        if len(eligible) < job.requested_gpu_count:
            return None
        return [gpu.id for gpu in eligible[: job.requested_gpu_count]]
