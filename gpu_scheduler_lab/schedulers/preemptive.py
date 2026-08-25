from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.job import Job
from gpu_scheduler_lab.schedulers.binpack import BinPackScheduler


class PreemptiveScheduler(BinPackScheduler):
    name = "preemptive"
    supports_preemption = True
    aging_interval = 30.0

    def pending_key(self, job: Job, now: float) -> tuple[float | int | str, ...]:
        return (-job.effective_priority(now, self.aging_interval), job.arrival_time, job.id)

    def place(self, cluster: Cluster, job: Job) -> list[str] | None:
        return super().place(cluster, job)
