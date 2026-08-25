from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gpu_scheduler_lab.admission.controller import AdmissionController
from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.fleet.events import FleetEventType
from gpu_scheduler_lab.metrics.fairness import jains_fairness_index
from gpu_scheduler_lab.metrics.fragmentation import fragmentation_snapshot
from gpu_scheduler_lab.metrics.summary import build_metrics
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.events import EVENT_ORDER, Event, EventType, TraceRecord
from gpu_scheduler_lab.models.job import Job, JobStatus
from gpu_scheduler_lab.models.topology import (
    topology_distance,
    topology_domain,
    topology_requirement_satisfied,
)
from gpu_scheduler_lab.schedulers.base import Scheduler
from gpu_scheduler_lab.schedulers.gang import validate_atomic_placement

if TYPE_CHECKING:
    from gpu_scheduler_lab.scenario import Scenario


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
                    "checkpoint_overhead": job.checkpoint_overhead,
                    "restart_overhead": job.restart_overhead,
                    "queue": job.queue_id,
                    "admission_time": job.admission_time,
                    "rejection_reason": job.rejection_reason,
                    "productive_work_completed": job.productive_work_completed,
                    "current_replicas": job.current_replicas,
                    "borrowed_gpu_units": job.borrowed_gpu_units,
                    "reclaim_victim_count": job.reclaim_victim_count,
                    "recovery_count": job.recovery_count,
                }
                for job in self.jobs
            ],
        }
        if include_trace:
            payload["trace"] = [record.to_dict() for record in self.trace]
        return payload


class Simulator:
    def __init__(
        self,
        cluster: Cluster,
        jobs: list[Job],
        scheduler: Scheduler,
        *,
        scenario: Scenario | None = None,
    ) -> None:
        self.cluster = cluster.clone()
        self.jobs = [job.clone() for job in jobs]
        self.scheduler = scheduler
        self.scenario = scenario
        self.accounting = scenario.accounting if scenario is not None else AccountingPolicy()
        self.admission = (
            AdmissionController(
                scheduler.hierarchy,
                self.cluster,
                self.accounting,
                scenario.admission_mode,
                {
                    event.node_id
                    for event in scenario.fleet_events
                    if event.event_type
                    in {
                        FleetEventType.NODE_JOIN,
                        FleetEventType.NODE_RECOVER,
                        FleetEventType.CAPACITY_RETURN,
                    }
                },
            )
            if scenario is not None and hasattr(scheduler, "hierarchy")
            else None
        )
        ids = [job.id for job in self.jobs]
        if len(set(ids)) != len(ids):
            raise ValueError("job ids must be unique")
        self.by_id = {job.id: job for job in self.jobs}
        self._elastic_jobs = [job for job in self.jobs if job.elastic is not None]
        self.pending: list[Job] = []
        self.running: dict[str, Job] = {}
        self.checkpointing: dict[str, Job] = {}
        self.restarting: dict[str, Job] = {}
        self._suspended_victims: dict[str, Job] = {}
        self._preemption_target_by_victim: dict[str, str] = {}
        self._preemption_reserved_gpus: dict[str, set[str]] = {}
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
        self._same_node_gang_placements = 0
        self._same_rack_gang_placements = 0
        self._cross_rack_gang_placements = 0
        self._cross_zone_gang_placements = 0
        self._topology_distance_sum = 0.0
        self._topology_distance_samples = 0
        self._topology_requirement_violations = 0
        self._gpu_capacity_area = 0.0
        self._memory_capacity_area = 0.0
        self._active_node_area = 0.0
        self._queue_gpu_area: dict[str, float] = {}
        self._queue_peak: dict[str, float] = {}
        self._queue_borrowed_area: dict[str, float] = {}
        self._queue_timeline: list[dict[str, Any]] = []
        self._elastic_timeline: list[dict[str, Any]] = []
        self._fleet_timeline: list[dict[str, Any]] = []
        self._rejection_reasons: dict[str, int] = {}
        self._fleet_counts = {event.value: 0 for event in FleetEventType}
        self._jobs_affected_by_capacity_loss: set[str] = set()
        self._revocable_gpu_time = 0.0
        self._stable_gpu_time = 0.0
        self._last_resize_time: dict[str, float] = {}
        self._elastic_replica_area: dict[str, float] = {}
        self._elastic_below_preferred: dict[str, float] = {}
        self._elastic_at_preferred: dict[str, float] = {}

    @classmethod
    def from_scenario(cls, scenario: Scenario, scheduler: Scheduler) -> Simulator:
        return cls(scenario.cluster, scenario.jobs, scheduler, scenario=scenario)

    def run(self) -> SimulationResult:
        started = time.perf_counter()
        for job in sorted(self.jobs, key=lambda item: (item.arrival_time, item.id)):
            self._push_event(job.arrival_time, EventType.JOB_ARRIVAL, job.id)
        if self.scenario is not None:
            for fleet_event in sorted(
                self.scenario.fleet_events,
                key=lambda item: (item.time, item.event_type.value, item.node_id),
            ):
                self._push_event(
                    fleet_event.time,
                    EventType(fleet_event.event_type.value),
                    fleet_event.node_id,
                )

        now = 0.0
        while self._events:
            self._discard_obsolete_ticks()
            self._discard_stale_generation_events()
            self._discard_aging_ticks_without_running_jobs()
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
                elif event.event_type is EventType.JOB_CHECKPOINT_COMPLETE:
                    self._complete_checkpoint(event, now)
                elif event.event_type is EventType.JOB_RESTART_COMPLETE:
                    self._complete_restart(event, now)
                elif event.event_type is EventType.JOB_ARRIVAL:
                    self._arrive(event, now)
                elif event.event_type is EventType.SCHEDULER_TICK:
                    self._handle_scheduler_tick(event, now)
                elif event.event_type in {
                    EventType.NODE_JOIN,
                    EventType.NODE_DRAIN,
                    EventType.NODE_FAIL,
                    EventType.NODE_RECOVER,
                    EventType.CAPACITY_REVOKE,
                    EventType.CAPACITY_RETURN,
                }:
                    self._handle_fleet_event(event, now)
            self._schedule(now)
            self._resize_elastic_jobs(now)
            self._record_phase3_timeline(now)
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
            node_count=len(self.cluster.schedulable_nodes),
            count_fragmentation_area=self._count_fragmentation_area,
            memory_fragmentation_area=self._memory_fragmentation_area,
            peak_gpu_utilization=self._peak_gpu_utilization,
            scheduling_attempts=self._scheduling_attempts,
            failed_attempts=self._failed_attempts,
            cross_node_gang_placements=self._cross_node_gang_placements,
            same_node_gang_placements=self._same_node_gang_placements,
            same_rack_gang_placements=self._same_rack_gang_placements,
            cross_rack_gang_placements=self._cross_rack_gang_placements,
            cross_zone_gang_placements=self._cross_zone_gang_placements,
            topology_distance_sum=self._topology_distance_sum,
            topology_distance_samples=self._topology_distance_samples,
            topology_requirement_violations=self._topology_requirement_violations,
            gpu_capacity_time=self._gpu_capacity_area,
            memory_capacity_time=self._memory_capacity_area,
            node_capacity_time=self._active_node_area,
        )
        metrics.update(self.scheduler.metrics())
        metrics.update(self._phase3_metrics(horizon))
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
        if self.admission is not None:
            decision = self.admission.decide(job)
            if not decision.admitted:
                job.status = JobStatus.REJECTED
                job.rejection_reason = decision.reason
                reason = decision.reason or "unspecified"
                self._rejection_reasons[reason] = self._rejection_reasons.get(reason, 0) + 1
                self.trace.append(
                    TraceRecord(now, EventType.JOB_REJECT, job.id, detail=f"reason={reason}")
                )
                return
            job.admission_time = now
            self.trace.append(TraceRecord(now, EventType.JOB_ADMIT, job.id))
        else:
            job.admission_time = now
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

    def _discard_stale_generation_events(self) -> None:
        expected_status = {
            EventType.JOB_COMPLETE: JobStatus.RUNNING,
            EventType.JOB_CHECKPOINT_COMPLETE: JobStatus.CHECKPOINTING,
            EventType.JOB_RESTART_COMPLETE: JobStatus.RESTARTING,
        }
        while self._events and self._events[0].event_type in expected_status:
            event = self._events[0]
            job = self.by_id[event.job_id]
            if (
                job.status is expected_status[event.event_type]
                and event.generation == job.run_generation
            ):
                break
            heapq.heappop(self._events)

    def _discard_aging_ticks_without_running_jobs(self) -> None:
        # A pending job that failed against an otherwise idle cluster cannot become
        # feasible through aging alone. Dropping its bookkeeping ticks keeps the
        # workload horizon independent of scheduler-internal timers.
        while (
            not self.running
            and not self.checkpointing
            and not self.restarting
            and self._events
            and self._events[0].event_type is EventType.SCHEDULER_TICK
        ):
            heapq.heappop(self._events)

    def _complete(self, event: Event, now: float) -> None:
        job = self.by_id[event.job_id]
        if job.status is not JobStatus.RUNNING or event.generation != job.run_generation:
            return
        if job.last_start_time is None:
            raise RuntimeError(f"running job {job.id} has no start time")
        self._accrue_productive_work(job, now)
        if job.elastic is None:
            job.accumulated_runtime = min(job.duration, job.accumulated_runtime)
        else:
            job.productive_work_completed = min(
                job.total_productive_work, job.productive_work_completed
            )
        gpu_ids = tuple(job.allocated_gpu_ids)
        node_ids = self._node_ids(gpu_ids)
        self.cluster.release(job)
        self.running.pop(job.id)
        job.status = JobStatus.COMPLETED
        job.completion_time = now
        job.last_start_time = None
        job.running_priority = None
        job.current_replicas = 0
        self.trace.append(TraceRecord(now, EventType.JOB_COMPLETE, job.id, gpu_ids, node_ids))

    def _schedule(self, now: float) -> None:
        allocated = [
            *self.running.values(),
            *self.checkpointing.values(),
            *self.restarting.values(),
        ]
        self.scheduler.prepare(now, self.cluster, self.pending, allocated)
        if self._placement_cannot_change_without_release():
            return
        self.pending.sort(key=lambda job: self.scheduler.pending_key(job, now))
        initial = tuple(self.pending)
        initial_ids = {job.id for job in initial}
        for job in initial:
            if self._placement_cannot_change_without_release():
                break
            if job.status is JobStatus.PENDING:
                self._attempt_placement(job, now, allow_preemption=True)

        # Victims added during this pass get one immediate resume opportunity, so
        # preemption cannot leave unrelated capacity idle until the next event.
        resumed_candidates = sorted(
            (job for job in self.pending if job.id not in initial_ids),
            key=lambda job: self.scheduler.pending_key(job, now),
        )
        for job in resumed_candidates:
            if self._placement_cannot_change_without_release():
                break
            if job.status is JobStatus.PENDING:
                self._attempt_placement(job, now, allow_preemption=False)

    def _placement_cannot_change_without_release(self) -> bool:
        return (
            not self.scheduler.supports_preemption
            and not self.scheduler.supports_reclaim
            and not any(gpu.free for gpu in self.cluster.schedulable_gpus)
        )

    def _attempt_placement(self, job: Job, now: float, *, allow_preemption: bool) -> bool:
        self._scheduling_attempts += 1
        placement = None
        requests = [job.gpu_count]
        if job.elastic is not None and self.scheduler.supports_elastic:
            requests = list(range(job.elastic.preferred_replicas, job.elastic.min_replicas - 1, -1))
        for replicas in requests:
            job.requested_replicas = replicas
            placement = self.scheduler.place(self._placement_cluster(job), job)
            if placement is not None:
                break
        if placement is None:
            self._failed_attempts += 1
            if allow_preemption and self.scheduler.supports_preemption:
                placement = self._preempt_for(job, now, reason="PREEMPT_PRIORITY")
            elif allow_preemption and self.scheduler.supports_reclaim:
                placement = self._preempt_for(job, now, reason="PREEMPT_RECLAIM")
        if placement is None:
            return False
        if not validate_atomic_placement(job, placement):
            raise RuntimeError(f"scheduler returned partial placement for {job.id}")
        self._start(job, placement, now)
        return True

    def _start(self, job: Job, placement: list[str], now: float) -> None:
        was_preempted = (
            job.preemption_count > 0
            or job.accumulated_runtime > 0
            or job.productive_work_completed > 0
            or job.recovery_count > 0
        )
        dispatch_priority = job.effective_priority(now, self.scheduler.aging_interval)
        self.cluster.allocate(job, placement)
        job.current_replicas = len(placement)
        self.pending.remove(job)
        job.running_priority = dispatch_priority
        job.first_start_time = now if job.first_start_time is None else job.first_start_time
        job.run_generation += 1
        node_ids = self._node_ids(tuple(placement))
        self._record_topology_placement(job, tuple(placement), node_ids)
        self._release_preemption_reservation(job.id, now)
        self.scheduler.on_job_started(job, now)
        if was_preempted and job.restart_cost > 0:
            job.status = JobStatus.RESTARTING
            job.last_start_time = None
            job.restart_overhead += job.restart_cost
            self.restarting[job.id] = job
            self.trace.append(
                TraceRecord(now, EventType.JOB_RESTART, job.id, tuple(placement), node_ids)
            )
            self._push_event(
                now + job.restart_cost,
                EventType.JOB_RESTART_COMPLETE,
                job.id,
                job.run_generation,
            )
            return
        job.status = JobStatus.RUNNING
        job.last_start_time = now
        self.running[job.id] = job
        event_type = EventType.JOB_RESUME if was_preempted else EventType.JOB_START
        self.trace.append(TraceRecord(now, event_type, job.id, tuple(placement), node_ids))
        self._schedule_completion(job, now)

    def _preempt_for(self, incoming: Job, now: float, *, reason: str) -> list[str] | None:
        incoming_priority = incoming.effective_priority(now, self.scheduler.aging_interval)
        if reason == "PREEMPT_RECLAIM" and self._shrink_elastic_for_reclaim(incoming, now):
            placement = self.scheduler.place(self._placement_cluster(incoming), incoming)
            if placement is not None:
                return placement
        victims = []
        for job in self.running.values():
            if not any(
                self.cluster.gpu_by_id(gpu_id).is_compatible(incoming)
                for gpu_id in job.allocated_gpu_ids
            ):
                continue
            if reason == "PREEMPT_RECLAIM":
                if self.scheduler.can_reclaim(job, incoming):
                    victims.append(job)
            elif (
                job.effective_priority(now, self.scheduler.aging_interval) < incoming_priority
                and job.priority < incoming.priority
            ):
                victims.append(job)
        victims.sort(
            key=lambda job: (
                job.effective_priority(now, self.scheduler.aging_interval),
                -self._suitable_gpu_count(job, incoming),
                len(job.allocated_gpu_ids),
                self._remaining_productive_runtime(job, now),
                job.checkpoint_cost + job.restart_cost,
                -job.borrowed_gpu_units,
                job.id,
            )
        )
        selected: list[Job] = []
        projected = self._placement_cluster(incoming).clone(preserve_allocations=True)
        projected_placement: list[str] | None = None
        for victim in victims:
            selected.append(victim)
            for gpu in projected.gpus:
                if gpu.owner_job_id == victim.id:
                    gpu.owner_job_id = None
                    gpu.allocated_memory_gb = 0.0
            projected_placement = self.scheduler.place(projected, incoming)
            if projected_placement is not None:
                break
        if projected_placement is None:
            return None
        defer_victims = any(victim.checkpoint_cost > 0 for victim in selected)
        if defer_victims:
            self._preemption_reserved_gpus[incoming.id] = set(projected_placement)
            for victim in selected:
                self._preemption_target_by_victim[victim.id] = incoming.id
        for victim in selected:
            self._begin_preemption(victim, now, incoming.id, reason)
        if defer_victims:
            return None
        self._scheduling_attempts += 1
        placement = self.scheduler.place(self._placement_cluster(incoming), incoming)
        if placement is None:
            self._failed_attempts += 1
        return placement

    def _suitable_gpu_count(self, victim: Job, incoming: Job) -> int:
        return sum(
            self.cluster.gpu_by_id(gpu_id).is_compatible(incoming)
            for gpu_id in victim.allocated_gpu_ids
        )

    def _begin_preemption(self, job: Job, now: float, incoming_id: str, reason: str) -> None:
        if job.last_start_time is None:
            raise RuntimeError(f"running job {job.id} has no start time")
        self._accrue_productive_work(job, now)
        gpu_ids = tuple(job.allocated_gpu_ids)
        node_ids = self._node_ids(gpu_ids)
        self.running.pop(job.id)
        job.last_start_time = None
        job.running_priority = None
        job.preemption_count += 1
        if reason == "PREEMPT_RECLAIM":
            job.reclaim_victim_count += 1
        job.checkpoint_overhead += job.checkpoint_cost
        if job.checkpoint_cost == 0:
            self.cluster.release(job)
            job.current_replicas = 0
            job.status = JobStatus.PENDING
            if job.id in self._preemption_target_by_victim:
                self._suspended_victims[job.id] = job
            else:
                self.pending.append(job)
                self._schedule_aging_tick(job, now)
            self.trace.append(
                TraceRecord(
                    now,
                    EventType.JOB_PREEMPT,
                    job.id,
                    gpu_ids,
                    node_ids,
                    detail=f"reason={reason};preempted_for={incoming_id}",
                )
            )
            return
        job.status = JobStatus.CHECKPOINTING
        self.checkpointing[job.id] = job
        self.trace.append(
            TraceRecord(
                now,
                EventType.JOB_PREEMPT,
                job.id,
                detail=f"reason={reason};preempted_for={incoming_id}",
            )
        )
        self._push_event(
            now + job.checkpoint_cost,
            EventType.JOB_CHECKPOINT_COMPLETE,
            job.id,
            job.run_generation,
        )

    def _complete_checkpoint(self, event: Event, now: float) -> None:
        job = self.by_id[event.job_id]
        if job.status is not JobStatus.CHECKPOINTING or event.generation != job.run_generation:
            return
        gpu_ids = tuple(job.allocated_gpu_ids)
        node_ids = self._node_ids(gpu_ids)
        self.cluster.release(job)
        job.current_replicas = 0
        self.checkpointing.pop(job.id)
        job.status = JobStatus.PENDING
        if job.id in self._preemption_target_by_victim:
            self._suspended_victims[job.id] = job
        else:
            self.pending.append(job)
            self._schedule_aging_tick(job, now)
        self.trace.append(
            TraceRecord(now, EventType.JOB_CHECKPOINT_COMPLETE, job.id, gpu_ids, node_ids)
        )

    def _complete_restart(self, event: Event, now: float) -> None:
        job = self.by_id[event.job_id]
        if job.status is not JobStatus.RESTARTING or event.generation != job.run_generation:
            return
        self.restarting.pop(job.id)
        job.status = JobStatus.RUNNING
        job.last_start_time = now
        self.running[job.id] = job
        self.trace.append(TraceRecord(now, EventType.JOB_RESTART_COMPLETE, job.id))
        self.trace.append(TraceRecord(now, EventType.JOB_RESUME, job.id))
        self._schedule_completion(job, now)

    def _schedule_completion(self, job: Job, now: float) -> None:
        if job.elastic is not None:
            remaining_work = max(0.0, job.total_productive_work - job.productive_work_completed)
            delay = remaining_work / job.productive_rate()
        else:
            delay = job.remaining_duration
        self._push_event(
            now + delay,
            EventType.JOB_COMPLETE,
            job.id,
            job.run_generation,
        )

    def _placement_cluster(self, job: Job) -> Cluster:
        reserved_elsewhere = {
            gpu_id
            for target, gpu_ids in self._preemption_reserved_gpus.items()
            if target != job.id
            for gpu_id in gpu_ids
        }
        if not reserved_elsewhere:
            return self.cluster
        projected = self.cluster.clone(preserve_allocations=True)
        for gpu_id in reserved_elsewhere:
            gpu = projected.gpu_by_id(gpu_id)
            if gpu.free:
                gpu.owner_job_id = "__preemption_reservation__"
        return projected

    def _release_preemption_reservation(self, incoming_id: str, now: float) -> None:
        if incoming_id not in self._preemption_reserved_gpus:
            return
        self._preemption_reserved_gpus.pop(incoming_id)
        victim_ids = [
            victim_id
            for victim_id, target in self._preemption_target_by_victim.items()
            if target == incoming_id
        ]
        for victim_id in victim_ids:
            self._preemption_target_by_victim.pop(victim_id)
            victim = self._suspended_victims.pop(victim_id, None)
            if victim is not None:
                self.pending.append(victim)
                self._schedule_aging_tick(victim, now)

    def _record_topology_placement(
        self, job: Job, gpu_ids: tuple[str, ...], node_ids: tuple[str, ...]
    ) -> None:
        unique = sorted(set(node_ids))
        nodes = {node.id: node for node in self.cluster.nodes}
        topologies = {node_id: nodes[node_id].topology for node_id in unique}
        if not topology_requirement_satisfied(job.topology_mode, unique, topologies):
            self._topology_requirement_violations += 1
        if job.gang:
            if len(unique) > 1:
                self._cross_node_gang_placements += 1
            if len(unique) == 1:
                self._same_node_gang_placements += 1
            else:
                racks = {
                    topology_domain(node_id, nodes[node_id].topology, "rack") for node_id in unique
                }
                zones = {
                    topology_domain(node_id, nodes[node_id].topology, "zone") for node_id in unique
                }
                if len(racks) == 1:
                    self._same_rack_gang_placements += 1
                elif len(zones) == 1:
                    self._cross_rack_gang_placements += 1
                else:
                    self._cross_zone_gang_placements += 1
        gpu_node_ids = [self.cluster.gpu_by_id(gpu_id).node_id for gpu_id in gpu_ids]
        distances = [
            topology_distance(
                left,
                nodes[left].topology,
                right,
                nodes[right].topology,
            )
            for index, left in enumerate(gpu_node_ids)
            for right in gpu_node_ids[index + 1 :]
        ]
        self._topology_distance_sum += sum(distances)
        self._topology_distance_samples += len(distances)

    def _handle_fleet_event(self, event: Event, now: float) -> None:
        node = next(node for node in self.cluster.nodes if node.id == event.job_id)
        self._fleet_counts[event.event_type.value] += 1
        if event.event_type is EventType.NODE_JOIN:
            node.available = True
            node.draining = False
            node.schedulable = True
        elif event.event_type is EventType.NODE_DRAIN:
            node.schedulable = False
            node.draining = True
        elif event.event_type in {EventType.NODE_FAIL, EventType.CAPACITY_REVOKE}:
            node.schedulable = False
            node.draining = False
            node.available = False
            self._capacity_loss(node.id, now, event.event_type)
        elif event.event_type in {
            EventType.NODE_RECOVER,
            EventType.CAPACITY_RETURN,
        }:
            node.available = True
            node.draining = False
            node.schedulable = True
        self.trace.append(
            TraceRecord(
                now,
                event.event_type,
                node.id,
                tuple(gpu.id for gpu in node.gpus),
                (node.id,),
            )
        )

    def _capacity_loss(self, node_id: str, now: float, event_type: EventType) -> None:
        node = next(node for node in self.cluster.nodes if node.id == node_id)
        impacted_ids = sorted(
            {gpu.owner_job_id for gpu in node.gpus if gpu.owner_job_id is not None}
        )
        for job_id in impacted_ids:
            if job_id not in self.by_id:
                continue
            job = self.by_id[job_id]
            self._jobs_affected_by_capacity_loss.add(job.id)
            unaffected = [
                gpu_id
                for gpu_id in job.allocated_gpu_ids
                if self.cluster.gpu_by_id(gpu_id).node_id != node_id
            ]
            if (
                job.elastic is not None
                and job.status is JobStatus.RUNNING
                and len(unaffected) >= job.elastic.min_replicas
            ):
                self._accrue_productive_work(job, now)
                job.last_start_time = now
                job.run_generation += 1
                old = job.current_replicas
                self.cluster.resize(job, unaffected)
                job.current_replicas = len(unaffected)
                job.requested_replicas = len(unaffected)
                job.elastic_scale_down_count += 1
                job.resize_churn_count += 1
                self.trace.append(
                    TraceRecord(
                        now,
                        EventType.ELASTIC_SCALE_DOWN,
                        job.id,
                        tuple(unaffected),
                        self._node_ids(tuple(unaffected)),
                        detail=f"reason={event_type.value};replicas={old}->{len(unaffected)}",
                    )
                )
                self._schedule_completion(job, now)
                continue
            self._interrupt_for_capacity_loss(job, now, event_type)

    def _interrupt_for_capacity_loss(self, job: Job, now: float, event_type: EventType) -> None:
        if job.status is JobStatus.RUNNING:
            self._accrue_productive_work(job, now)
        self.running.pop(job.id, None)
        self.checkpointing.pop(job.id, None)
        self.restarting.pop(job.id, None)
        if job.allocated_gpu_ids:
            self.cluster.release(job)
        job.current_replicas = 0
        job.last_start_time = None
        job.running_priority = None
        job.run_generation += 1
        job.recovery_count += 1
        job.recovery_overhead += job.restart_cost
        job.status = JobStatus.PENDING
        if job not in self.pending:
            if job.id in self._preemption_target_by_victim:
                self._suspended_victims[job.id] = job
            else:
                self.pending.append(job)
        self.trace.append(
            TraceRecord(
                now,
                EventType.JOB_PREEMPT,
                job.id,
                detail=f"reason=PREEMPT_CAPACITY_REVOKE;event={event_type.value}",
            )
        )

    def _resize_elastic_jobs(self, now: float) -> None:
        if not self.scheduler.supports_elastic or not self._elastic_jobs:
            return
        for job in sorted(self.running.values(), key=lambda item: item.id):
            if (
                job.elastic is None
                or job.current_replicas >= job.elastic.preferred_replicas
                or self._last_resize_time.get(job.id) == now
                or not self.scheduler.can_scale_up(job)
            ):
                continue
            target_replicas = next(
                (
                    replicas
                    for replicas in range(
                        job.elastic.preferred_replicas,
                        job.current_replicas,
                        -1,
                    )
                    if self.scheduler.can_resize(job, replicas)
                ),
                job.current_replicas,
            )
            needed = target_replicas - job.current_replicas
            if needed <= 0:
                continue
            free = sorted(
                self.cluster.eligible_gpus(job),
                key=lambda gpu: (gpu.node_id, gpu.id),
            )
            if len(free) < needed:
                continue
            self._accrue_productive_work(job, now)
            target = [*job.allocated_gpu_ids, *(gpu.id for gpu in free[:needed])]
            old = job.current_replicas
            self.cluster.resize(job, target)
            job.current_replicas = len(target)
            job.requested_replicas = len(target)
            job.elastic_scale_up_count += 1
            job.resize_churn_count += 1
            job.run_generation += 1
            job.last_start_time = now
            self._last_resize_time[job.id] = now
            self.trace.append(
                TraceRecord(
                    now,
                    EventType.ELASTIC_SCALE_UP,
                    job.id,
                    tuple(target),
                    self._node_ids(tuple(target)),
                    detail=f"replicas={old}->{len(target)}",
                )
            )
            self._schedule_completion(job, now)
            break

    def _shrink_elastic_for_reclaim(self, incoming: Job, now: float) -> bool:
        changed = False
        candidates = sorted(
            (
                job
                for job in self.running.values()
                if job.elastic is not None
                and job.current_replicas > job.elastic.min_replicas
                and self.scheduler.can_reclaim(job, incoming)
            ),
            key=lambda job: (-job.borrowed_gpu_units, job.id),
        )
        for job in candidates:
            assert job.elastic is not None
            self._accrue_productive_work(job, now)
            old = job.current_replicas
            target = sorted(job.allocated_gpu_ids)[: job.elastic.min_replicas]
            self.cluster.resize(job, target)
            job.current_replicas = len(target)
            job.requested_replicas = len(target)
            job.borrowed_gpu_units = 0.0
            job.elastic_scale_down_count += 1
            job.resize_churn_count += 1
            job.run_generation += 1
            job.last_start_time = now
            self._last_resize_time[job.id] = now
            self.trace.append(
                TraceRecord(
                    now,
                    EventType.ELASTIC_SCALE_DOWN,
                    job.id,
                    tuple(target),
                    self._node_ids(tuple(target)),
                    detail=f"reason=PREEMPT_RECLAIM;replicas={old}->{len(target)}",
                )
            )
            self._schedule_completion(job, now)
            changed = True
            if len(self.cluster.eligible_gpus(incoming)) >= incoming.requested_gpu_count:
                break
        return changed

    def _record_phase3_timeline(self, now: float) -> None:
        if hasattr(self.scheduler, "queue_snapshot") and hasattr(self.scheduler, "refresh_usage"):
            allocated = [
                *self.running.values(),
                *self.checkpointing.values(),
                *self.restarting.values(),
            ]
            self.scheduler.refresh_usage(self.cluster, allocated)
            self._append_timeline(
                self._queue_timeline, {"time": now, "queues": self.scheduler.queue_snapshot()}
            )
        if self._elastic_jobs:
            self._append_timeline(
                self._elastic_timeline,
                {
                    "time": now,
                    "replicas": {job.id: job.current_replicas for job in self._elastic_jobs},
                },
            )
        self._append_timeline(
            self._fleet_timeline,
            {
                "time": now,
                "schedulable_gpus": self.cluster.total_gpu_count,
                "active_gpus": len(self.cluster.active_gpus),
                "revocable_gpus": sum(
                    len(node.gpus) for node in self.cluster.schedulable_nodes if node.revocable
                ),
            },
        )

    @staticmethod
    def _append_timeline(timeline: list[dict[str, Any]], point: dict[str, Any]) -> None:
        if len(timeline) >= 1024:
            timeline[:] = timeline[::2]
        timeline.append(point)

    def _phase3_metrics(self, horizon: float) -> dict[str, Any]:
        admitted = [job for job in self.jobs if job.admission_time is not None]
        rejected = [job for job in self.jobs if job.status is JobStatus.REJECTED]
        admission_waits = [
            admission_time - job.arrival_time
            for job in admitted
            if (admission_time := job.admission_time) is not None
        ]
        queue_waits = [
            job.first_start_time - job.admission_time
            for job in admitted
            if job.first_start_time is not None and job.admission_time is not None
        ]
        queue_metrics: dict[str, dict[str, Any]] = {}
        if hasattr(self.scheduler, "hierarchy") and hasattr(self.scheduler, "queue_snapshot"):
            hierarchy = self.scheduler.hierarchy
            snapshot = self.scheduler.queue_snapshot()
            for queue_id, spec in sorted(hierarchy.specs.items()):
                queue_jobs = [
                    job for job in self.jobs if queue_id in hierarchy.ancestors(job.queue_id)
                ]
                average = self._queue_gpu_area.get(queue_id, 0.0) / horizon if horizon else 0.0
                guarantee = spec.guaranteed.gpu_units
                waits = [
                    job.first_start_time - job.admission_time
                    for job in queue_jobs
                    if job.first_start_time is not None and job.admission_time is not None
                ]
                service_quality = self._queue_service_quality(queue_jobs)
                queue_metrics[queue_id] = {
                    "guaranteed_gpu_units": guarantee,
                    "max_gpu_units": spec.limit.gpu_units if spec.limit is not None else None,
                    "average_gpu_usage": average,
                    "peak_gpu_usage": self._queue_peak.get(queue_id, 0.0),
                    "borrowed_gpu_time": self._queue_borrowed_area.get(queue_id, 0.0),
                    "guaranteed_share_satisfaction": (
                        min(1.0, average / guarantee) if guarantee else 1.0
                    ),
                    "admission_wait": 0.0,
                    "scheduling_wait": sum(waits) / len(waits) if waits else 0.0,
                    "completed_jobs": sum(job.status is JobStatus.COMPLETED for job in queue_jobs),
                    "rejected_jobs": sum(job.status is JobStatus.REJECTED for job in queue_jobs),
                    "preemption_count": sum(job.preemption_count for job in queue_jobs),
                    "reclaim_victim_count": sum(job.reclaim_victim_count for job in queue_jobs),
                    "historical_service": snapshot[queue_id]["historical_service"],
                    "normalized_entitlement": snapshot[queue_id]["normalized_entitlement"],
                    "fairshare_debt": snapshot[queue_id]["fairshare_debt"],
                    "sla_violation_rate": self._queue_sla_violation_rate(queue_jobs),
                    "service_quality": service_quality,
                }
        elastic_jobs = self._elastic_jobs
        running_time = sum(
            self._elastic_below_preferred.get(job.id, 0.0)
            + self._elastic_at_preferred.get(job.id, 0.0)
            for job in elastic_jobs
        )
        starvation_threshold = (
            self.scenario.starvation_threshold if self.scenario is not None else 300.0
        )
        starvation_count = sum(
            (job.first_start_time if job.first_start_time is not None else horizon)
            - (job.admission_time if job.admission_time is not None else job.arrival_time)
            > starvation_threshold
            for job in admitted
        )
        leaves = self.scheduler.hierarchy.leaves() if hasattr(self.scheduler, "hierarchy") else ()
        service_leaves = [
            queue_id for queue_id in leaves if any(job.queue_id == queue_id for job in self.jobs)
        ]
        satisfaction = [
            queue_metrics[queue_id]["guaranteed_share_satisfaction"] for queue_id in service_leaves
        ]
        satisfaction_mean = sum(satisfaction) / len(satisfaction) if satisfaction else 0.0
        satisfaction_variance = (
            sum((value - satisfaction_mean) ** 2 for value in satisfaction) / len(satisfaction)
            if satisfaction
            else 0.0
        )
        metrics: dict[str, Any] = {
            "submitted_job_count": len(self.jobs),
            "admitted_job_count": len(admitted),
            "rejected_job_count": len(rejected),
            "average_admission_wait_time": (
                sum(admission_waits) / len(admission_waits) if admission_waits else 0.0
            ),
            "average_queue_wait_time": sum(queue_waits) / len(queue_waits) if queue_waits else 0.0,
            "admission_wait_time": (
                sum(admission_waits) / len(admission_waits) if admission_waits else 0.0
            ),
            "queue_wait_time": sum(queue_waits) / len(queue_waits) if queue_waits else 0.0,
            "rejection_reason_counts": dict(sorted(self._rejection_reasons.items())),
            "queue_metrics": queue_metrics,
            "starvation_count": starvation_count,
            "guarantee_satisfaction_variance": satisfaction_variance,
            "queue_service_jains_index": jains_fairness_index(
                [queue_metrics[queue_id]["service_quality"] for queue_id in service_leaves]
            ),
            "elastic_job_count": len(elastic_jobs),
            "elastic_scale_up_count": sum(job.elastic_scale_up_count for job in elastic_jobs),
            "elastic_scale_down_count": sum(job.elastic_scale_down_count for job in elastic_jobs),
            "average_allocated_replicas": (
                sum(self._elastic_replica_area.values()) / running_time if running_time else 0.0
            ),
            "time_below_preferred": sum(self._elastic_below_preferred.values()),
            "time_at_or_above_preferred": sum(self._elastic_at_preferred.values()),
            "elastic_work_completed": sum(job.productive_work_completed for job in elastic_jobs),
            "resize_churn_count": sum(job.resize_churn_count for job in elastic_jobs),
            "node_join_count": self._fleet_counts[EventType.NODE_JOIN.value],
            "node_drain_count": self._fleet_counts[EventType.NODE_DRAIN.value],
            "node_failure_count": self._fleet_counts[EventType.NODE_FAIL.value],
            "capacity_revoke_count": self._fleet_counts[EventType.CAPACITY_REVOKE.value],
            "capacity_return_count": self._fleet_counts[EventType.CAPACITY_RETURN.value],
            "jobs_affected_by_capacity_loss": len(self._jobs_affected_by_capacity_loss),
            "recovery_count": sum(job.recovery_count for job in self.jobs),
            "recovery_overhead": sum(job.recovery_overhead for job in self.jobs),
            "revocable_gpu_time": self._revocable_gpu_time,
            "stable_gpu_time": self._stable_gpu_time,
            "queue_timeline": self._queue_timeline,
            "elastic_replica_timeline": self._elastic_timeline,
            "fleet_capacity_timeline": self._fleet_timeline,
        }
        self._assert_finite_metrics(metrics)
        return metrics

    @staticmethod
    def _queue_sla_violation_rate(jobs: list[Job]) -> float:
        sla_jobs = [job for job in jobs if job.sla_deadline is not None]
        if not sla_jobs:
            return 0.0
        violations = sum(
            job.completion_time is None or job.completion_time > job.sla_deadline  # type: ignore[operator]
            for job in sla_jobs
        )
        return violations / len(sla_jobs)

    @staticmethod
    def _queue_service_quality(jobs: list[Job]) -> float:
        demand = sum(job.total_productive_work for job in jobs)
        if demand == 0:
            return 0.0
        completed = sum(
            job.total_productive_work for job in jobs if job.status is JobStatus.COMPLETED
        )
        turnaround_work = sum(
            job.turnaround_time * job.preferred_gpu_count
            for job in jobs
            if job.status is JobStatus.COMPLETED and job.turnaround_time is not None
        )
        completion_ratio = completed / demand
        latency_efficiency = min(1.0, completed / turnaround_work) if turnaround_work else 0.0
        return completion_ratio * latency_efficiency

    @staticmethod
    def _assert_finite_metrics(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("metrics must not contain NaN or Infinity")
        if isinstance(value, dict):
            for item in value.values():
                Simulator._assert_finite_metrics(item)
        elif isinstance(value, list):
            for item in value:
                Simulator._assert_finite_metrics(item)

    def _integrate_state(self, delta: float) -> None:
        if delta <= 0:
            return
        busy = sum(gpu.occupied for gpu in self.cluster.active_gpus)
        allocated_memory = sum(gpu.allocated_memory_gb for gpu in self.cluster.active_gpus)
        active_nodes = sum(
            any(gpu.occupied for gpu in node.gpus) for node in self.cluster.schedulable_nodes
        )
        count_fragmentation, memory_fragmentation, _ = fragmentation_snapshot(self.cluster)
        self._busy_gpu_time += busy * delta
        self._gpu_capacity_area += len(self.cluster.active_gpus) * delta
        self._memory_capacity_area += (
            sum(gpu.memory_capacity_gb for gpu in self.cluster.active_gpus) * delta
        )
        self._active_node_area += len(self.cluster.active_nodes) * delta
        self._revocable_gpu_time += (
            sum(len(node.gpus) for node in self.cluster.active_nodes if node.revocable) * delta
        )
        self._stable_gpu_time += (
            sum(len(node.gpus) for node in self.cluster.active_nodes if not node.revocable) * delta
        )
        self._memory_area += allocated_memory * delta
        self._node_area += active_nodes * delta
        self._count_fragmentation_area += count_fragmentation * delta
        self._memory_fragmentation_area += memory_fragmentation * delta
        if self.cluster.active_gpus:
            self._peak_gpu_utilization = max(
                self._peak_gpu_utilization, busy / len(self.cluster.active_gpus)
            )
        if hasattr(self.scheduler, "queue_snapshot"):
            snapshot = self.scheduler.queue_snapshot()
            for queue_id, values in snapshot.items():
                usage = values["gpu_units"]
                borrowed = values["borrowed_usage"]
                self._queue_gpu_area[queue_id] = (
                    self._queue_gpu_area.get(queue_id, 0.0) + usage * delta
                )
                self._queue_borrowed_area[queue_id] = (
                    self._queue_borrowed_area.get(queue_id, 0.0) + borrowed * delta
                )
                self._queue_peak[queue_id] = max(self._queue_peak.get(queue_id, 0.0), usage)
        for job in self.running.values():
            if job.elastic is None:
                continue
            self._elastic_replica_area[job.id] = (
                self._elastic_replica_area.get(job.id, 0.0) + job.current_replicas * delta
            )
            if job.current_replicas < job.elastic.preferred_replicas:
                self._elastic_below_preferred[job.id] = (
                    self._elastic_below_preferred.get(job.id, 0.0) + delta
                )
            else:
                self._elastic_at_preferred[job.id] = (
                    self._elastic_at_preferred.get(job.id, 0.0) + delta
                )

    def _node_ids(self, gpu_ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({self.cluster.gpu_by_id(gpu_id).node_id for gpu_id in gpu_ids}))

    @staticmethod
    def _runtime_so_far(job: Job, now: float) -> float:
        active = now - job.last_start_time if job.last_start_time is not None else 0.0
        return job.accumulated_runtime + active

    @staticmethod
    def _remaining_productive_runtime(job: Job, now: float) -> float:
        if job.elastic is not None:
            active = 0.0
            if job.last_start_time is not None:
                active = (now - job.last_start_time) * job.productive_rate()
            remaining = max(0.0, job.total_productive_work - job.productive_work_completed - active)
            return remaining / max(job.productive_rate(), 1e-9)
        return max(0.0, job.duration - Simulator._runtime_so_far(job, now))

    @staticmethod
    def _accrue_productive_work(job: Job, now: float) -> None:
        if job.last_start_time is None:
            return
        elapsed = max(0.0, now - job.last_start_time)
        if job.elastic is None:
            job.accumulated_runtime += elapsed
            job.productive_work_completed = min(
                job.total_productive_work,
                job.accumulated_runtime * job.gpu_count,
            )
        else:
            job.productive_work_completed = min(
                job.total_productive_work,
                job.productive_work_completed + elapsed * job.productive_rate(),
            )
