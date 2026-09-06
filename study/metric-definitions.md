# Canonical metric definitions

All metrics come from deterministic simulator output. Ratios use the simulator's
logical horizon and modeled schedulable capacity; they are not measurements of a
production scheduler, GPU kernel, network, or storage system.

| ID | Source key | Definition | Direction |
|---|---|---|---|
| `average-gpu-utilization` | `average_gpu_utilization` | Busy GPU capacity time divided by schedulable GPU capacity time. | maximize |
| `p95-wait` | `p95_waiting_time` | 95th percentile of completed-job queue delay, defined uniformly as first start time minus arrival time for fixed and elastic jobs. | minimize |
| `completion-rate` | `completion_rate` | Completed jobs divided by submitted jobs at the horizon. | maximize |
| `gpu-count-fragmentation` | `gpu_count_fragmentation` | Time-weighted free GPU slots unusable for pending count demands. | minimize |
| `gpu-memory-fragmentation` | `gpu_memory_fragmentation` | Time-weighted free memory unusable for pending placement demands. | minimize |
| `guaranteed-share-satisfaction` | `queue_metrics.*.guaranteed_share_satisfaction` | Satisfied entitled demand area divided by entitled demand area for each queue. | maximize |
| `jain-service-quality-fairness` | `queue_service_jains_index` | Jain index over active leaf-queue service quality. | maximize |
| `preemption-overhead` | `preemption_overhead_ratio` | Modeled wasted checkpoint/restart GPU time divided by busy GPU time. | minimize |
| `checkpoint-overhead` | `total_checkpoint_overhead` | Sum of modeled checkpoint time. | minimize |
| `restart-overhead` | `total_restart_overhead` | Sum of modeled restart time. | minimize |
| `average-topology-distance` | `average_topology_distance` | Mean simulator topology cost across sampled placements. | minimize |
| `topology-violation` | `topology_requirement_violation_count` | Count of placements violating a declared topology requirement. | minimize |
| `sla-violation` | `sla_violation_rate` | SLA jobs unfinished or late divided by SLA jobs. | minimize |

Aggregation across seeds must retain sample count and dispersion. A later report may
describe observed differences, but it must not translate logical time into real
hardware latency or claim causal significance without a justified method.
