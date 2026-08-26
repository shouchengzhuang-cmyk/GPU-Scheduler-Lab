# Public trace study

This repository keeps the normal fixture-based trace checks small and deterministic. The optional
B7 evidence path uses the complete public Alibaba `cluster-trace-v2026-spot-gpu` source files for
input provenance and normalization coverage, then runs frozen, bounded replay windows through the
discrete-event simulator.

## Evidence semantics

The output is always labeled `SIMULATED_TRACE_REPLAY`. It is not evidence from a real NVIDIA GPU,
Kubernetes cluster, vLLM server, or Alibaba production scheduler. The public CSV files are never
committed to this repository, and the result bundle contains only source hashes, normalization
counts, frozen configuration, aggregate simulator metrics, a report, and bundle hashes.

## Reproduce locally

```bash
python -m pip install -e '.[dev]'
python scripts/download_trace.py --output-dir .data/alibaba-spot-gpu-v2026
python scripts/run_public_trace_study.py \
  --input .data/alibaba-spot-gpu-v2026 \
  --config study/public-trace-study.yaml \
  --output-dir build/public-trace-study
cd build/public-trace-study
sha256sum -c hashes.sha256
```

The downloader records SHA-256 for every source CSV. The study runner refuses to continue if the
files do not match the manifest or if the manifest dataset version does not match the frozen study
configuration.

## What is original and what is synthetic

Original source fields include node identity, GPU model/capacity, CPU capacity, job identity,
organization, resource requests, worker count, submit time, duration, and HP/Spot class. The
adapter derives simulator arrival time, aggregate integer GPU count, priority, and GPU-memory
capacity from frozen mappings.

The source `organization` field is used as tenant identity, but the queue path, queue weight, and
equal GPU-unit guarantee are synthetic experiment controls. They exist only to exercise the
fair-share and reclaim policies consistently. They must never be described as Alibaba production
configuration.

Fractional GPU rows are counted and excluded because the current simulator allocates integer GPU
units. GPU models without an explicit memory-capacity mapping are also counted and excluded rather
than assigned guessed capacity. The report records both exclusions.

## Frozen replay contract

`study/public-trace-study.yaml` pins the dataset version, allowed GPU-memory mappings, three window
quantiles, 24-hour logical windows, replay bounds, seed, policies, tenant-overlay strategy, and
claim boundary. Changing any of those inputs changes the study contract and should be reviewed as
a methodology change.

The three windows are anchored at deterministic quantiles of distinct eligible submit times. The
source files are ingested in full for hashing and normalization statistics, while each simulator
window is bounded so the experiment remains affordable and repeatable in GitHub Actions.

## GitHub Actions

`.github/workflows/public-trace-study.yml` is intentionally separate from normal CI. It is manually
dispatchable and, while B7 is under development, also runs on pushes to the dedicated B7 branch.
The workflow downloads the public trace, runs the frozen study, verifies `hashes.sha256`, checks the
evidence marker, and uploads the result bundle as a workflow artifact.
