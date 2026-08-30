from __future__ import annotations

import heapq
import math
import time
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gpu_scheduler_lab.admission.controller import AdmissionController
from gpu_scheduler_lab.fairshare.accounting import AccountingPolicy
from gpu_scheduler_lab.fleet.events import FleetEventType, schedulable_node_snapshots
from gpu_scheduler_lab.metrics.fairness import jains_fairness_index
from gpu_scheduler_lab.metrics.fragmentation import fragmentation_snapshot
from gpu_scheduler_lab.metrics.summary import build_metrics
from gpu_scheduler_lab.models.accelerator import AcceleratorVendor
from gpu_scheduler_lab.models.cluster import Cluster
from gpu_scheduler_lab.models.events import EVENT_ORDER, Event, EventType, TraceRecord
from gpu_scheduler_lab.models.job import Job, JobStatus
from gpu_scheduler_lab.models.topology import (
    topology_distance,
    topology_domain,
    topology_requirement_satisfied,
)
from gpu_scheduler_lab.queues.model import ResourceVector
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
        self._preemption_reservation_reason: dict[str, str] = {}
        self._preemption_reclaim_entitlements: dict[str, set[str]] = {}
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
        self._queue_entitled_demand_area: dict[str, float] = {}
        self._queue_satisfied_entitlement_area: dict[str, float] = {}
        self._direct_runnable_demand: dict[str, ResourceVector] = {}
        self._runnable_demand_by_job: dict[str, ResourceVector] = {}
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
            while self._events:
                event_count = len(self._events)
                self._discard_obsolete_ticks()
                self._discard_stale_generation_events()
                self._discard_aging_ticks_without_running_jobs()
                if len(self._events) == event_count:
                    break
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
        capacity_snapshots = self._capacity_snapshots(now)
        if self.admission is not None:
            decision = self.admission.decide(job, capacity_snapshots)
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
        self._add_runnable_demand(job, capacity_snapshots)
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
        self._remove_runnable_demand(job)
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
        if self.scheduler.supports_guarantee_placement:
            self._dispatch_pending(
                initial,
                now,
                guaranteed_only=True,
                allow_preemption=False,
            )
            allocated = [
                *self.running.values(),
                *self.checkpointing.values(),
                *self.restarting.values(),
            ]
            self.scheduler.prepare(now, self.cluster, self.pending, allocated)
        self._dispatch_pending(
            tuple(self.pending),
            now,
            guaranteed_only=False,
            allow_preemption=True,
        )

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

    def _dispatch_pending(
        self,
        candidates: tuple[Job, ...],
        now: float,
        *,
        guaranteed_only: bool,
        allow_preemption: bool,
    ) -> None:
        remaining = list(candidates)
        while remaining:
            if self._placement_cannot_change_without_release():
                break
            if self.scheduler.dynamic_pending_order:
                job = min(remaining, key=lambda item: self.scheduler.pending_key(item, now))
                remaining.remove(job)
            else:
                job = remaining.pop(0)
            if job.status is JobStatus.PENDING:
                self._attempt_placement(
                    job,
                    now,
                    allow_preemption=allow_preemption,
                    guaranteed_only=guaranteed_only,
                )

    def _placement_cannot_change_without_release(self) -> bool:
        return (
            not self.scheduler.supports_preemption
            and not self.scheduler.supports_reclaim
            and not any(gpu.free for gpu in self.cluster.schedulable_gpus)
        )

    def _attempt_placement(
        self,
        job: Job,
        now: float,
        *,
        allow_preemption: bool,
        guaranteed_only: bool = False,
    ) -> bool:
        self._scheduling_attempts += 1
        placement = None
        reclaim_reservation = self._preemption_reservation_reason.get(job.id) == "PREEMPT_RECLAIM"
        reclaim_entitlements = self._preemption_reclaim_entitlements.get(job.id, set())
        requests = [job.gpu_count]
        if job.elastic is not None and self.scheduler.supports_elastic:
            requests = [job.elastic.min_replicas]
        place = self.scheduler.place_guaranteed if guaranteed_only else self.scheduler.place
        for replicas in requests:
            job.requested_replicas = replicas
            placement = place(self._placement_cluster(job), job)
            if (
                placement is not None
                and reclaim_reservation
                and not self.scheduler.can_reclaim_placement(
                    job,
                    placement,
                    reclaim_entitlements,
                )
            ):
                placement = None
                continue
            if placement is not None:
                break
        if reclaim_reservation and placement is None:
            reserved = self._preemption_reserved_gpus.get(job.id, set())
            if all(self.cluster.gpu_by_id(gpu_id).free for gpu_id in reserved):
                self._release_preemption_reservation(job.id, now)
            self._failed_attempts += 1
            return False
        if guaranteed_only and placement is None:
            return False
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
        if reason == "PREEMPT_RECLAIM":
            return self._reclaim_for(incoming, now)
        incoming_priority = incoming.effective_priority(now, self.scheduler.aging_interval)
        eligible_victims = [
            job
            for job in self.running.values()
            if any(
                self.cluster.gpu_by_id(gpu_id).is_compatible(incoming)
                for gpu_id in job.allocated_gpu_ids
            )
            and (
                job.effective_priority(now, self.scheduler.aging_interval) < incoming_priority
                and job.priority < incoming.priority
            )
        ]

        def victim_key(job: Job) -> tuple[Any, ...]:
            return (
                job.effective_priority(now, self.scheduler.aging_interval),
                -self._suitable_gpu_count(job, incoming),
                len(job.allocated_gpu_ids),
                self._remaining_productive_runtime(job, now),
                job.checkpoint_cost + job.restart_cost,
                -job.borrowed_gpu_units,
                job.id,
            )

        placement_cluster = self._placement_cluster(incoming)
        compatible_vendors = sorted(
            {
                gpu.vendor
                for gpu in placement_cluster.schedulable_gpus
                if gpu.is_compatible(incoming)
            },
            key=lambda vendor: vendor.value,
        )
        plans: list[tuple[tuple[Any, ...], list[Job], list[str]]] = []
        for vendor in compatible_vendors:
            projected = placement_cluster.clone(preserve_allocations=True, vendor=vendor)
            victims = sorted(
                (
                    job
                    for job in eligible_victims
                    if any(
                        self.cluster.gpu_by_id(gpu_id).vendor is vendor
                        for gpu_id in job.allocated_gpu_ids
                    )
                ),
                key=victim_key,
            )
            selected: list[Job] = []
            projected_placement = self.scheduler.place(projected, incoming)
            for victim in victims:
                if projected_placement is not None:
                    break
                selected.append(victim)
                for gpu in projected.gpus:
                    if gpu.owner_job_id == victim.id:
                        gpu.owner_job_id = None
                        gpu.allocated_memory_gb = 0.0
                projected_placement = self.scheduler.place(projected, incoming)
            if projected_placement is None:
                continue
            plans.append(
                (
                    (
                        sum(job.checkpoint_cost + job.restart_cost for job in selected),
                        len(selected),
                        sum(len(job.allocated_gpu_ids) for job in selected),
                        tuple(victim_key(job) for job in selected),
                        vendor.value,
                    ),
                    selected,
                    projected_placement,
                )
            )
        if not plans:
            return None
        _, selected, projected_placement = min(plans, key=lambda plan: plan[0])
        defer_victims = any(victim.checkpoint_cost > 0 for victim in selected)
        if defer_victims:
            self._preemption_reserved_gpus[incoming.id] = set(projected_placement)
            self._preemption_reservation_reason[incoming.id] = reason
            for victim in selected:
                self._preemption_target_by_victim[victim.id] = incoming.id
        for victim in selected:
            self._begin_preemption(victim, now, incoming.id, reason)
        if defer_victims:
            return None
        allocated = [
            *self.running.values(),
            *self.checkpointing.values(),
            *self.restarting.values(),
        ]
        self.scheduler.prepare(now, self.cluster, self.pending, allocated)
        self._scheduling_attempts += 1
        placement = self.scheduler.place(self._placement_cluster(incoming), incoming)
        if placement is None:
            self._failed_attempts += 1
        return placement

    def _reclaim_for(self, incoming: Job, now: float) -> list[str] | None:
        if not self.scheduler.can_reclaim_request(incoming):
            return None
        actual_allocated = [
            *self.running.values(),
            *self.checkpointing.values(),
            *self.restarting.values(),
        ]
        projected = self._placement_cluster(incoming).clone(preserve_allocations=True)
        projected_jobs: dict[str, Job] = {}
        for job in actual_allocated:
            projected_job = copy(job)
            projected_job.allocated_gpu_ids = list(job.allocated_gpu_ids)
            projected_jobs[job.id] = projected_job
        allocated_order = [job.id for job in actual_allocated]
        planned_targets: dict[str, list[str]] = {}
        preempted_ids: list[str] = []
        entitlement_queue_ids: set[str] = set()

        def projected_allocated() -> list[Job]:
            preempted = set(preempted_ids)
            return [projected_jobs[job_id] for job_id in allocated_order if job_id not in preempted]

        def refresh_projected() -> None:
            self.scheduler.prepare(now, projected, self.pending, projected_allocated())

        def target_placement() -> list[str] | None:
            placement = self.scheduler.place(projected, incoming)
            if placement is None or not self.scheduler.can_reclaim_placement(
                incoming,
                placement,
                entitlement_queue_ids,
            ):
                return None
            return placement

        refresh_projected()
        projected_placement: list[str] | None = None
        while projected_placement is None:
            elastic_actions: list[tuple[tuple[Any, ...], str, str, str]] = []
            for job in self.running.values():
                projected_job = projected_jobs[job.id]
                if (
                    projected_job.elastic is None
                    or projected_job.current_replicas <= projected_job.elastic.min_replicas
                    or not self.scheduler.can_reclaim(projected_job, incoming)
                ):
                    continue
                entitlement = self.scheduler.reclaim_entitlement_queue(projected_job, incoming)
                if entitlement is None:
                    continue
                for gpu_id in projected_job.allocated_gpu_ids:
                    gpu = projected.gpu_by_id(gpu_id)
                    if not gpu.is_compatible(incoming) or not self.scheduler.can_reclaim_allocation(
                        projected_job,
                        incoming,
                        [gpu_id],
                    ):
                        continue
                    elastic_actions.append(
                        (
                            (
                                projected_job.effective_priority(
                                    now, self.scheduler.aging_interval
                                ),
                                -self.accounting.model_weights.get(gpu.model, 1.0),
                                self._remaining_productive_runtime(job, now),
                                job.checkpoint_cost + job.restart_cost,
                                job.id,
                                gpu_id,
                            ),
                            job.id,
                            gpu_id,
                            entitlement,
                        )
                    )
            if not elastic_actions:
                break
            _, job_id, gpu_id, entitlement = min(elastic_actions, key=lambda item: item[0])
            projected_job = projected_jobs[job_id]
            target = [item for item in projected_job.allocated_gpu_ids if item != gpu_id]
            planned_targets[job_id] = target
            projected_job.allocated_gpu_ids = list(target)
            projected_job.current_replicas = len(target)
            gpu = projected.gpu_by_id(gpu_id)
            gpu.owner_job_id = None
            gpu.allocated_memory_gb = 0.0
            entitlement_queue_ids.add(entitlement)
            refresh_projected()
            projected_placement = target_placement()

        while projected_placement is None:
            victim_actions: list[tuple[tuple[Any, ...], str, str]] = []
            for job in self.running.values():
                if job.id in preempted_ids:
                    continue
                projected_job = projected_jobs[job.id]
                released = list(projected_job.allocated_gpu_ids)
                if (
                    not released
                    or not any(
                        projected.gpu_by_id(gpu_id).is_compatible(incoming) for gpu_id in released
                    )
                    or not self.scheduler.can_reclaim(projected_job, incoming)
                    or not self.scheduler.can_reclaim_allocation(
                        projected_job,
                        incoming,
                        released,
                        allow_indivisible_collateral=True,
                    )
                ):
                    continue
                entitlement = self.scheduler.reclaim_entitlement_queue(projected_job, incoming)
                if entitlement is None:
                    continue
                suitable = sum(
                    projected.gpu_by_id(gpu_id).is_compatible(incoming) for gpu_id in released
                )
                victim_actions.append(
                    (
                        (
                            projected_job.effective_priority(now, self.scheduler.aging_interval),
                            -suitable,
                            len(released),
                            self._remaining_productive_runtime(job, now),
                            job.checkpoint_cost + job.restart_cost,
                            -job.borrowed_gpu_units,
                            job.id,
                        ),
                        job.id,
                        entitlement,
                    )
                )
            if not victim_actions:
                break
            _, job_id, entitlement = min(victim_actions, key=lambda item: item[0])
            projected_job = projected_jobs[job_id]
            for gpu_id in projected_job.allocated_gpu_ids:
                gpu = projected.gpu_by_id(gpu_id)
                gpu.owner_job_id = None
                gpu.allocated_memory_gb = 0.0
            projected_job.allocated_gpu_ids = []
            projected_job.current_replicas = 0
            planned_targets.pop(job_id, None)
            preempted_ids.append(job_id)
            entitlement_queue_ids.add(entitlement)
            refresh_projected()
            projected_placement = target_placement()

        self.scheduler.prepare(now, self.cluster, self.pending, actual_allocated)
        if projected_placement is None:
            return None

        for job_id, target in sorted(planned_targets.items()):
            job = self.running[job_id]
            if target == job.allocated_gpu_ids:
                continue
            self._accrue_productive_work(job, now)
            old = job.current_replicas
            self.cluster.resize(job, target)
            job.current_replicas = len(target)
            job.requested_replicas = len(target)
            job.elastic_scale_down_count += 1
            job.resize_churn_count += 1
            job.run_generation += 1
            job.last_start_time = now
            self._last_resize_time[job.id] = now
            gpu_ids = tuple(target)
            self.trace.append(
                TraceRecord(
                    now,
                    EventType.ELASTIC_SCALE_DOWN,
                    job.id,
                    gpu_ids,
                    self._node_ids(gpu_ids),
                    detail=f"reason=PREEMPT_RECLAIM;replicas={old}->{len(target)}",
                )
            )
            self._schedule_completion(job, now)

        selected = [self.running[job_id] for job_id in preempted_ids]
        defer_victims = any(victim.checkpoint_cost > 0 for victim in selected)
        if defer_victims:
            self._preemption_reserved_gpus[incoming.id] = set(projected_placement)
            self._preemption_reservation_reason[incoming.id] = "PREEMPT_RECLAIM"
            self._preemption_reclaim_entitlements[incoming.id] = set(entitlement_queue_ids)
            for victim in selected:
                self._preemption_target_by_victim[victim.id] = incoming.id
        for victim in selected:
            self._begin_preemption(victim, now, incoming.id, "PREEMPT_RECLAIM")
        if defer_victims:
            return None

        allocated = [
            *self.running.values(),
            *self.checkpointing.values(),
            *self.restarting.values(),
        ]
        self.scheduler.prepare(now, self.cluster, self.pending, allocated)
        placement = self.scheduler.place(self._placement_cluster(incoming), incoming)
        if placement is None or not self.scheduler.can_reclaim_placement(
            incoming,
            placement,
            entitlement_queue_ids,
        ):
            raise RuntimeError("committed reclaim plan did not preserve target placement")
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
        self._preemption_reservation_reason.pop(incoming_id, None)
        self._preemption_reclaim_entitlements.pop(incoming_id, None)
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

    def _invalidate_reservations_for_node(self, node_id: str, now: float) -> None:
        node_gpu_ids = {
            gpu.id for node in self.cluster.nodes if node.id == node_id for gpu in node.gpus
        }
        affected_targets = sorted(
            target_id
            for target_id, gpu_ids in self._preemption_reserved_gpus.items()
            if gpu_ids.intersection(node_gpu_ids)
        )
        for target_id in affected_targets:
            self._release_preemption_reservation(target_id, now)

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
            self._invalidate_reservations_for_node(node.id, now)
        elif event.event_type in {EventType.NODE_FAIL, EventType.CAPACITY_REVOKE}:
            node.schedulable = False
            node.draining = False
            node.available = False
            self._invalidate_reservations_for_node(node.id, now)
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
        blocked = {
            job_id for job_id, resized_at in self._last_resize_time.items() if resized_at == now
        }
        original_replicas: dict[str, int] = {}
        while True:
            allocated = [
                *self.running.values(),
                *self.checkpointing.values(),
                *self.restarting.values(),
            ]
            self.scheduler.prepare(now, self.cluster, self.pending, allocated)
            candidates = sorted(
                (
                    job
                    for job in self.running.values()
                    if job.elastic is not None
                    and job.id not in blocked
                    and job.current_replicas < job.elastic.preferred_replicas
                    and self.scheduler.can_scale_up(job)
                ),
                key=lambda job: self.scheduler.scale_up_key(job, now),
            )
            resized = False
            for job in candidates:
                target = self._elastic_resize_target(job, job.current_replicas + 1)
                if target is None or not self.scheduler.can_resize_placement(job, target):
                    continue
                if job.id not in original_replicas:
                    self._accrue_productive_work(job, now)
                    original_replicas[job.id] = job.current_replicas
                self.cluster.resize(job, target)
                job.current_replicas = len(target)
                job.requested_replicas = len(target)
                job.last_start_time = now
                resized = True
                break
            if not resized:
                break
        for job_id, old in sorted(original_replicas.items()):
            job = self.running[job_id]
            job.elastic_scale_up_count += 1
            job.resize_churn_count += 1
            job.run_generation += 1
            self._last_resize_time[job.id] = now
            gpu_ids = tuple(job.allocated_gpu_ids)
            node_ids = self._node_ids(gpu_ids)
            self._record_topology_placement(job, gpu_ids, node_ids)
            self.trace.append(
                TraceRecord(
                    now,
                    EventType.ELASTIC_SCALE_UP,
                    job.id,
                    gpu_ids,
                    node_ids,
                    detail=f"replicas={old}->{job.current_replicas}",
                )
            )
            self._schedule_completion(job, now)

    def _elastic_resize_target(self, job: Job, replicas: int) -> list[str] | None:
        needed = replicas - job.current_replicas
        if needed <= 0:
            return None
        reserved = {
            gpu_id
            for target_id, gpu_ids in self._preemption_reserved_gpus.items()
            if target_id != job.id
            for gpu_id in gpu_ids
        }
        free = sorted(
            (gpu for gpu in self.cluster.eligible_gpus(job) if gpu.id not in reserved),
            key=lambda gpu: (
                self.accounting.model_weights.get(gpu.model, 1.0),
                gpu.node_id,
                gpu.id,
            ),
        )
        nodes = {node.id: node for node in self.cluster.nodes}
        topologies = {node_id: node.topology for node_id, node in nodes.items()}
        selected: list[str] = []
        for gpu in free:
            gpu_ids = [*job.allocated_gpu_ids, *selected, gpu.id]
            node_ids = [self.cluster.gpu_by_id(gpu_id).node_id for gpu_id in gpu_ids]
            if not topology_requirement_satisfied(job.topology_mode, node_ids, topologies):
                continue
            selected.append(gpu.id)
            if len(selected) == needed:
                return [*job.allocated_gpu_ids, *selected]
        return None

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
        active_gpus = self.cluster.active_gpus
        revocable_node_ids = {node.id for node in self.cluster.nodes if node.revocable}
        self._append_timeline(
            self._fleet_timeline,
            {
                "time": now,
                "schedulable_gpus": self.cluster.total_gpu_count,
                "active_gpus": len(active_gpus),
                "revocable_gpus": sum(gpu.node_id in revocable_node_ids for gpu in active_gpus),
            },
        )

    @staticmethod
    def _append_timeline(timeline: list[dict[str, Any]], point: dict[str, Any]) -> None:
        if len(timeline) >= 1024:
            timeline[:] = timeline[::2]
        timeline.append(point)

    def _capacity_snapshots(self, now: float) -> tuple[frozenset[str], ...]:
        fleet_events = self.scenario.fleet_events if self.scenario is not None else ()
        return schedulable_node_snapshots(self.cluster, fleet_events, after=now)

    def _add_runnable_demand(
        self,
        job: Job,
        node_snapshots: tuple[frozenset[str], ...],
    ) -> None:
        candidates: list[tuple[int, ResourceVector]] = []
        for node_ids in node_snapshots:
            compatible = [
                gpu
                for node in self.cluster.nodes
                if node.id in node_ids
                for gpu in node.gpus
                if gpu.is_compatible(job)
            ]
            compatible_by_vendor: dict[AcceleratorVendor, int] = {}
            for gpu in compatible:
                compatible_by_vendor[gpu.vendor] = compatible_by_vendor.get(gpu.vendor, 0) + 1
            replicas = max(
                (
                    min(job.preferred_gpu_count, vendor_capacity)
                    for vendor_capacity in compatible_by_vendor.values()
                ),
                default=0,
            )
            if replicas < job.minimum_gpu_count:
                continue
            try:
                demand = self.accounting.minimum_demand(job, compatible, replicas)
            except ValueError:
                continue
            candidates.append((replicas, demand))
        if not candidates:
            return
        replicas = max(candidate[0] for candidate in candidates)
        demand = min(
            (candidate[1] for candidate in candidates if candidate[0] == replicas),
            key=lambda item: (item.gpu_units, item.gpu_memory_gb),
        )
        self._runnable_demand_by_job[job.id] = demand
        self._direct_runnable_demand[job.queue_id] = (
            self._direct_runnable_demand.get(job.queue_id, ResourceVector()) + demand
        )

    def _remove_runnable_demand(self, job: Job) -> None:
        demand = self._runnable_demand_by_job.pop(job.id, None)
        if demand is None:
            return
        remaining = self._direct_runnable_demand.get(job.queue_id, ResourceVector()) - demand
        if remaining == ResourceVector():
            self._direct_runnable_demand.pop(job.queue_id, None)
        else:
            self._direct_runnable_demand[job.queue_id] = remaining

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
                entitled_demand_area = self._queue_entitled_demand_area.get(queue_id, 0.0)
                satisfied_entitlement_area = self._queue_satisfied_entitlement_area.get(
                    queue_id, 0.0
                )
                queue_metrics[queue_id] = {
                    "guaranteed_gpu_units": guarantee,
                    "max_gpu_units": spec.limit.gpu_units if spec.limit is not None else None,
                    "average_gpu_usage": average,
                    "peak_gpu_usage": self._queue_peak.get(queue_id, 0.0),
                    "borrowed_gpu_time": self._queue_borrowed_area.get(queue_id, 0.0),
                    "guaranteed_share_satisfaction": (
                        min(1.0, satisfied_entitlement_area / entitled_demand_area)
                        if entitled_demand_area
                        else 1.0
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
        active_gpus = self.cluster.active_gpus
        revocable_node_ids = {node.id for node in self.cluster.nodes if node.revocable}
        busy = sum(gpu.occupied for gpu in active_gpus)
        allocated_memory = sum(gpu.allocated_memory_gb for gpu in active_gpus)
        active_nodes = sum(
            any(gpu.occupied for gpu in node.gpus) for node in self.cluster.active_nodes
        )
        count_fragmentation, memory_fragmentation, _ = fragmentation_snapshot(self.cluster)
        self._busy_gpu_time += busy * delta
        self._gpu_capacity_area += len(active_gpus) * delta
        self._memory_capacity_area += sum(gpu.memory_capacity_gb for gpu in active_gpus) * delta
        self._active_node_area += len(self.cluster.active_nodes) * delta
        self._revocable_gpu_time += (
            sum(gpu.node_id in revocable_node_ids for gpu in active_gpus) * delta
        )
        self._stable_gpu_time += (
            sum(gpu.node_id not in revocable_node_ids for gpu in active_gpus) * delta
        )
        self._memory_area += allocated_memory * delta
        self._node_area += active_nodes * delta
        self._count_fragmentation_area += count_fragmentation * delta
        self._memory_fragmentation_area += memory_fragmentation * delta
        if active_gpus:
            self._peak_gpu_utilization = max(self._peak_gpu_utilization, busy / len(active_gpus))
        if hasattr(self.scheduler, "hierarchy") and hasattr(self.scheduler, "queue_snapshot"):
            snapshot = self.scheduler.queue_snapshot()
            hierarchy = self.scheduler.hierarchy
            aggregate_demand = hierarchy.aggregate_usage(self._direct_runnable_demand)
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
                spec = hierarchy.specs[queue_id]
                if "gpu_units" in (spec.guaranteed_dimensions or ()):
                    entitled_demand = min(
                        spec.guaranteed.gpu_units,
                        aggregate_demand[queue_id].gpu_units,
                    )
                    self._queue_entitled_demand_area[queue_id] = (
                        self._queue_entitled_demand_area.get(queue_id, 0.0)
                        + entitled_demand * delta
                    )
                    self._queue_satisfied_entitlement_area[queue_id] = (
                        self._queue_satisfied_entitlement_area.get(queue_id, 0.0)
                        + min(usage, entitled_demand) * delta
                    )
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
