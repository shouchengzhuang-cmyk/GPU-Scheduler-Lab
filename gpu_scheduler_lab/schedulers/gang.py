from gpu_scheduler_lab.models.job import Job


def validate_atomic_placement(job: Job, placement: list[str] | None) -> bool:
    """All jobs are atomic; gang jobs make this contract explicit in traces/metrics."""
    return placement is None or len(placement) == job.gpu_count
