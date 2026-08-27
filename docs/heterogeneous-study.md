# Heterogeneous NVIDIA and Huawei Ascend study

## Scope

The heterogeneous study reuses the existing `Scenario -> Simulator -> Scheduler -> Metrics` path. It adds explicit accelerator feasibility and evidence governance; it does not emulate CUDA, CANN, vLLM, device plugins, or hardware performance.

Typed placement checks:

- vendor: `nvidia` or `huawei-ascend`;
- kind: `gpu` or `npu`;
- model and memory fit;
- required capabilities;
- runtime profile;
- topology, quota, fair share, reclaim, fragmentation, and current capacity;
- one vendor per multi-device gang.

`prefer-nvidia` and `prefer-ascend` are deterministic placement preferences. They first try the preferred vendor and then perform a bounded fallback to another explicitly compatible vendor. They never weaken a task's vendor/kind/model/profile/capability constraints and never create a mixed-vendor gang.

## Modes

### Correctness

Correctness mode compares only simulator-level feasibility and scheduling outcomes. It may report utilization, memory fit, topology, quota, fair-share, reclaim, fragmentation, vendor restrictions, and capacity loss. It must not report a hardware performance winner.

```bash
python -m gpu_scheduler_lab heterogeneous-study \
  --config experiments/heterogeneous-correctness.yaml
```

The canonical fixture runs both route preferences against the baseline and two time-zero vendor outage ablations.

### Calibrated

Calibrated mode accepts explicit profiles with all of these fields:

```yaml
performance_profiles:
  - source_kind: MEASURED | ASSUMED | SYNTHETIC
    source_id: stable-source-identifier
    model_variant: model-and-physical-variant
    ttft_ms: 100
    tpot_ms: 10
    throughput_tokens_s: 100
    power_watts: 300
    cost_per_hour: 1
```

Every profile must declare `source_kind`. If the compared set is not entirely `MEASURED`, the manifest sets `performance_comparison.status=NOT_PERMITTED`; raw values remain visible but no vendor ranking is generated.

```bash
python -m gpu_scheduler_lab heterogeneous-study \
  --config experiments/heterogeneous-calibrated-synthetic.yaml
```

The bundled values are deliberately `SYNTHETIC`, equal, and illustrative. They are not benchmark results.

## Artifacts and evidence boundary

Each run writes:

- `manifest.json`: exact scenario hash, typed inventory, v2 contract check, profile evidence kinds, route/outage variables, and real-hardware boundary;
- `runs.json`: deterministic correctness outcomes without wall-clock benchmark values;
- `report.md`: separate Facts, Assumptions, Synthetic variables, correctness results, performance evidence boundary, and real-hardware boundary.

The Mini AI Cloud v2 fixture is parsed through the standalone adapter on every study run. No result is written back to Mini AI Cloud.

Current hardware evidence:

- NVIDIA: `REAL_HW_NOT_RUN`;
- Huawei Ascend: `REAL_HW_NOT_RUN`;
- deployment: `NOT_DEPLOYED`.
