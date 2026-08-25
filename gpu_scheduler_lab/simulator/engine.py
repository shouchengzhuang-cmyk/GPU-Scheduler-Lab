from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any

from gpu_scheduler_lab.metrics.fragmentation import fragmentation_snapshot
from gpu_scheduler_lab.metrics.summary import build_metrics
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.events import EVENT_ORDER, Event, EventType, TraceRecord
from gpu_scheduler_lab.models.job import Job, JobStatus
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.schedulers.gang import validate_atomic_placement


@dataclass(slots=True)
class SimulationResult:
    scheduler: str
    metrics: dict[str, Any]
    jobs: list[Job]
    trace: list[TraceRecord]
    elapsed_seconds: float

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scheduler": self.scheduler,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
            "jobs": [
                {
                    "id": job.id,
                    "status": job.status.value,
                    "arrival_time": job.arrival_time,
                    "first_start_time": job.first_start_time,
                    "completion_time": job.completion_time,
                    "waiting_time": job.waiting_time,
                    "turnaround_time": job.turnaround_time,
                    "preemption_count": job.preemption_count,
                    "accumulated_runtime": job.accumulated_runtime,
                }
                for job in self.jobs
            ],
        }
        if include_trace:
            payload["trace"] = [record.to_dict() for record in self.trace]
        return payload


class Simulator:
    def __init__(self, cluster: Cluster, jobs: list[Job], scheduler: Scheduler) -> None:
        self.cluster = cluster.clone()
        self.jobs = [job.clone() for job in jobs]
        self.scheduler = scheduler
        ids = [job.id for job in self.jobs]
        if len(set(ids)) != len(ids):
            raise ValueError("job ids must be unique")
        self.by_id = {job.id: job for job in self.jobs}
        self.pending: list[Job] = []
        self.running: dict[str, Job] = {}
        self.trace: list[TraceRecord] = []
        self._events: list[Event] = []
        self._sequence = 0
        self._busy_gpu_time = 0.0
        self._memory_area = 0.0
        self._node_area = 0.0
        self._count_fragmentation_area = 0.0
        self._memory_fragmentation_area = 0.0
        self._peak_gpu_utilization = 0.0
        self._scheduling_attempts = 0
        self._failed_attempts = 0
        self._cross_node_gang_placements = 0

    def run(self) -> SimulationResult:
        started = time.perf_counter()
        for job in sorted(self.jobs, key=lambda item: (item.arrival_time, item.id)):
            self._push_event(job.arrival_time, EventType.JOB_ARRIVAL, job.id)

        now = 0.0
        while self._events:
            self._discard_obsolete_ticks()
            if not self._events:
                break
            event_time = self._events[0].time
            self._integrate_state(event_time - now)
            now = event_time
            current: list[Event] = []
            while self._events and self._events[0].time == event_time:
                current.append(heapq.heappop(self._events))
            current.sort()
            for event in current:
                if event.event_type is EventType.JOB_COMPLETE:
                    self._complete(event, now)
                elif event.event_type is EventType.JOB_ARRIVAL:
                    self._arrive(event, now)
                elif event.event_type is EventType.SCHEDULER_TICK:
                    self._handle_scheduler_tick(event, now)
            self._schedule(now)
            self.cluster.assert_invariants()

        horizon = now
        metrics = build_metrics(
            self.jobs,
            horizon=horizon,
            total_gpus=self.cluster.total_gpu_count,
            busy_gpu_time=self._busy_gpu_time,
            memory_area=self._memory_area,
            total_memory_gb=self.cluster.total_memory_gb,
            node_area=self._node_area,
            node_count=len(self.cluster.nodes),
            count_fragmentation_area=self._count_fragmentation_area,
            memory_fragmentation_area=self._memory_fragmentation_area,
            peak_gpu_utilization=self._peak_gpu_utilization,
            scheduling_attempts=self._scheduling_attempts,
            failed_attempts=self._failed_attempts,
            cross_node_gang_placements=self._cross_node_gang_placements,
        )
        return SimulationResult(
            scheduler=self.scheduler.name,
            metrics=metrics,
            jobs=self.jobs,
            trace=self.trace,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _push_event(
        self,
        at: float,
        event_type: EventType,
        job_id: str,
        generation: int = 0,
    ) -> None:
        self._sequence += 1
        heapq.heappush(
            self._events,
            Event(
                time=at,
                order=EVENT_ORDER[event_type],
                sequence=self._sequence,
                event_type=event_type,
                job_id=job_id,
                generation=generation,
            ),
        )

    def _arrive(self, event: Event, now: float) -> None:
        job = self.by_id[event.job_id]
        self.pending.append(job)
        self.trace.append(TraceRecord(now, EventType.JOB_ARRIVAL, job.id))
        self._schedule_aging_tick(job, now)

    def _handle_scheduler_tick(self, event: Event, now: float) -> None:
        job = self.by_id[event.job_id]
        if job in self.pending:
            self._schedule_aging_tick(job, now)

    def _schedule_aging_tick(self, job: Job, now: float) -> None:
        if not self.scheduler.supports_preemption:
            return
        if job.effective_priority(now, self.scheduler.aging_interval) >= 3:
            return
        self._push_event(
            now + self.scheduler.aging_interval,
            EventType.SCHEDULER_TICK,
            job.id,
        )

    def _discard_obsolete_ticks(self) -> None:
        while self._events and self._events[0].event_type is EventType.SCHEDULER_TICK:
            job = self.by_id[self._events[0].job_id]
            if job in self.pending:
                break
            heapq.heappop(self._events)

    def _complete(self, event: Event, now: float) -> None:
        job = self.by_id[event.job_id]
        if job.status is not JobStatus.RUNNING or event.generation != job.run_generation:
            return
        if job.last_start_time is None:
            raise RuntimeError(f"running job {job.id} has no start time")
        job.accumulated_runtime = min(
            job.duration, job.accumulated_runtime + now - job.last_start_time
        )
        gpu_ids = tuple(job.allocated_gpu_ids)
        node_ids = self._node_ids(gpu_ids)
        self.cluster.release(job)
        self.running.pop(job.id)
        job.status = JobStatus.COMPLETED
        job.completion_time = now
        job.last_start_time = None
        job.running_priority = None
        self.trace.append(TraceRecord(now, EventType.JOB_COMPLETE, job.id, gpu_ids, node_ids))

    def _schedule(self, now: float) -> None:
        self.pending.sort(key=lambda job: self.scheduler.pending_key(job, now))
        initial = tuple(self.pending)
        initial_ids = {job.id for job in initial}
        for job in initial:
            if job.status is JobStatus.PENDING:
                self._attempt_placement(job, now, allow_preemption=True)

        # Victims added during this pass get one immediate resume opportunity, so
        # preemption cannot leave unrelated capacity idle until the next event.
        resumed_candidates = sorted(
            (job for job in self.pending if job.id not in initial_ids),
            key=lambda job: self.scheduler.pending_key(job, now),
        )
        for job in resumed_candidates:
            if job.status is JobStatus.PENDING:
                self._attempt_placement(job, now, allow_preemption=False)

    def _attempt_placement(self, job: Job, now: float, *, allow_preemption: bool) -> bool:
        self._scheduling_attempts += 1
        placement = self.scheduler.place(self.cluster, job)
        if placement is None:
            self._failed_attempts += 1
            if allow_preemption and self.scheduler.supports_preemption:
                placement = self._preempt_for(job, now)
        if placement is None:
            return False
        if not validate_atomic_placement(job, placement):
            raise RuntimeError(f"scheduler returned partial placement for {job.id}")
        self._start(job, placement, now)
        return True

    def _start(self, job: Job, placement: list[str], now: float) -> None:
        was_preempted = job.preemption_count > 0 or job.accumulated_runtime > 0
        dispatch_priority = job.effective_priority(now, self.scheduler.aging_interval)
        self.cluster.allocate(job, placement)
        self.pending.remove(job)
        job.status = JobStatus.RUNNING
        job.last_start_time = now
        job.running_priority = dispatch_priority
        job.first_start_time = now if job.first_start_time is None else job.first_start_time
        job.run_generation += 1
        self.running[job.id] = job
        node_ids = self._node_ids(tuple(placement))
        if job.gang and len(node_ids) > 1:
            self._cross_node_gang_placements += 1
        event_type = EventType.JOB_RESUME if was_preempted else EventType.JOB_START
        self.trace.append(TraceRecord(now, event_type, job.id, tuple(placement), node_ids))
        self._push_event(
            now + job.remaining_duration,
            EventType.JOB_COMPLETE,
            job.id,
            job.run_generation,
        )

    def _preempt_for(self, incoming: Job, now: float) -> list[str] | None:
        incoming_priority = incoming.effective_priority(now, self.scheduler.aging_interval)
        victims = [
            job
            for job in self.running.values()
            if job.effective_priority(now, self.scheduler.aging_interval) < incoming_priority
            and job.priority < incoming.priority
            and any(
                self.cluster.gpu_by_id(gpu_id).memory_capacity_gb >= incoming.gpu_memory_gb
                for gpu_id in job.allocated_gpu_ids
            )
        ]
        victims.sort(
            key=lambda job: (
                job.effective_priority(now, self.scheduler.aging_interval),
                -len(job.allocated_gpu_ids),
                self._runtime_so_far(job, now),
                job.id,
            )
        )
        free_suitable = len(self.cluster.eligible_gpus(incoming.gpu_memory_gb))
        selected: list[Job] = []
        recoverable = free_suitable
        for victim in victims:
            selected.append(victim)
            recoverable += sum(
                self.cluster.gpu_by_id(gpu_id).memory_capacity_gb >= incoming.gpu_memory_gb
                for gpu_id in victim.allocated_gpu_ids
            )
            if recoverable >= incoming.gpu_count:
                break
        if recoverable < incoming.gpu_count:
            return None
        for victim in selected:
            self._preempt(victim, now, incoming.id)
        self._scheduling_attempts += 1
        placement = self.scheduler.place(self.cluster, incoming)
        if placement is None:
            self._failed_attempts += 1
        return placement

    def _preempt(self, job: Job, now: float, incoming_id: str) -> None:
        if job.last_start_time is None:
            raise RuntimeError(f"running job {job.id} has no start time")
        job.accumulated_runtime += now - job.last_start_time
        gpu_ids = tuple(job.allocated_gpu_ids)
        node_ids = self._node_ids(gpu_ids)
        self.cluster.release(job)
        self.running.pop(job.id)
        job.status = JobStatus.PENDING
        job.last_start_time = None
        job.running_priority = None
        job.preemption_count += 1
        self.pending.append(job)
        self._schedule_aging_tick(job, now)
        self.trace.append(
            TraceRecord(
                now,
                EventType.JOB_PREEMPT,
                job.id,
                gpu_ids,
                node_ids,
                detail=f"preempted_for={incoming_id}",
            )
        )

    def _integrate_state(self, delta: float) -> None:
        if delta <= 0:
            return
        busy = sum(gpu.occupied for gpu in self.cluster.gpus)
        allocated_memory = sum(gpu.allocated_memory_gb for gpu in self.cluster.gpus)
        active_nodes = sum(any(gpu.occupied for gpu in node.gpus) for node in self.cluster.nodes)
        count_fragmentation, memory_fragmentation, _ = fragmentation_snapshot(self.cluster)
        self._busy_gpu_time += busy * delta
        self._memory_area += allocated_memory * delta
        self._node_area += active_nodes * delta
        self._count_fragmentation_area += count_fragmentation * delta
        self._memory_fragmentation_area += memory_fragmentation * delta
        if self.cluster.total_gpu_count:
            self._peak_gpu_utilization = max(
                self._peak_gpu_utilization, busy / self.cluster.total_gpu_count
            )

    def _node_ids(self, gpu_ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({self.cluster.gpu_by_id(gpu_id).node_id for gpu_id in gpu_ids}))

    @staticmethod
    def _runtime_so_far(job: Job, now: float) -> float:
        active = now - job.last_start_time if job.last_start_time is not None else 0.0
        return job.accumulated_runtime + active
