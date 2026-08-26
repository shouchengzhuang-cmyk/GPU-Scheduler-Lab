# Canonical study hypotheses

The canonical study asks how the four frozen policies trade utilization, latency,
fairness, fragmentation, and reliability under five controlled variables. These
hypotheses define comparisons to run; they are not conclusions.

| ID | Hypothesis | Independent variable | Dependent metrics |
|---|---|---|---|
| `h1-load-pressure` | Higher offered load increases tail wait and can reduce completion. | `workload-intensity` | `p95-wait`, `completion-rate`, `average-gpu-utilization` |
| `h2-heterogeneity-fragmentation` | A more heterogeneous fleet increases count or memory fragmentation. | `gpu-heterogeneity` | `gpu-count-fragmentation`, `gpu-memory-fragmentation`, `average-gpu-utilization` |
| `h3-topology-tradeoff` | Stricter locality reduces distance at a latency or completion cost. | `topology-strictness` | `average-topology-distance`, `topology-violation`, `p95-wait`, `completion-rate` |
| `h4-recovery-cost` | Higher checkpoint/restart cost increases modeled preemption overhead. | `checkpoint-restart-cost` | `preemption-overhead`, `checkpoint-overhead`, `restart-overhead`, `completion-rate` |
| `h5-revocable-fairness` | More revocable capacity stresses guarantees, fairness, and SLA outcomes. | `revocable-capacity-ratio` | `guaranteed-share-satisfaction`, `jain-service-quality-fairness`, `sla-violation`, `restart-overhead` |

All independent and dependent IDs are validated against `study.yaml`. A missing or
unknown ID is a configuration error. Later experiment PRs may test these hypotheses;
this milestone intentionally does not report a result or statistical claim.
