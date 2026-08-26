from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.schedulers.base import Scheduler


class FIFOScheduler(Scheduler):
    name = "fifo"

    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        eligible = cluster.eligible_gpus(job.gpu_memory_gb)
        if len(eligible) < job.gpu_count:
            return None
        return [gpu.id for gpu in eligible[: job.gpu_count]]
