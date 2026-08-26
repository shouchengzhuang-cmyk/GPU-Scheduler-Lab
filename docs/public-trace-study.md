# Public trace study

This repository keeps normal fixture-based checks small and deterministic. The optional B7 path
uses the complete public Alibaba `cluster-trace-v2026-spot-gpu` inputs for provenance and
normalization coverage, then runs frozen bounded replay windows through the discrete-event
simulator.

## Evidence semantics

Every result is labeled `SIMULATED_TRACE_REPLAY`. It is not evidence from a real NVIDIA GPU,
Kubernetes cluster, vLLM server, Alibaba production scheduler, or production control plane. The
raw upstream README and CSV files are not committed. The result bundle contains source identity
and hashes, normalization counts, frozen configuration, aggregate simulator metrics, a report,
and bundle hashes.

## Pinned source and attribution

B7 pins Alibaba `alibaba/clusterdata` at commit
`c08f563115af39bad047353431bf745b4dee665c`. The downloader fetches the upstream `README.md`,
`node_info_df.csv`, and `job_info_df.csv` from that exact revision and records SHA-256 for all
three. The pinned README travels only in the ignored local data directory; its hash is retained in
the evidence manifest so the attribution/citation context can be identified later.

The upstream directory does not provide this project with an independent data license to
relicense. Review the pinned upstream README, its paper citation, and current upstream terms before
reusing or redistributing the dataset. This repository does not redistribute the full trace.

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

The study runner refuses to continue if the manifest dataset version or pinned upstream revision
differs from the frozen study config, if a required file is missing, or if any downloaded file
fails SHA-256 verification.

## What is original and what is synthetic

Original source fields include node identity, GPU model/capacity, CPU capacity, job identity,
organization, resource requests, worker count, submit time, duration, and HP/Spot class. The
adapter derives simulator arrival time, aggregate integer GPU count, priority, and GPU-memory
capacity from the frozen mappings.

The source `organization` field is used only as tenant identity. Queue path, queue weight, and equal
GPU-unit guarantee are synthetic experiment controls. They exist to exercise fair-share and
reclaim policies consistently and must never be described as Alibaba production configuration.

The public node file exposes six labels, including anonymized `GPU-series-1` and `GPU-series-2`.
B7 deliberately does not guess memory capacity for those anonymized labels. The frozen map covers
A10, A100-SXM4-80GB, A800-SXM4-80GB, and H800; rows outside that auditable map are counted as
model exclusions. Fractional/non-integer GPU rows are also counted and excluded because the
simulator allocates integer GPU units. Invalid and duplicate rows are reported separately.

## Frozen replay contract

`study/public-trace-study.yaml` pins the upstream commit, dataset version, four explicit GPU-memory
mappings, three window quantiles, 24-hour logical windows, replay bounds, seed, policies,
tenant-overlay strategy, and claim boundary. Changing any of these inputs is a methodology change.

The three windows are anchored at deterministic quantiles of distinct eligible submit times. The
source files are ingested in full for hashing and normalization statistics, while each simulator
window is bounded so the experiment remains affordable and repeatable.

## GitHub Actions

`.github/workflows/public-trace-study.yml` is intentionally separate from normal fixture CI. It is
manually dispatchable and also validates relevant B7 branch/PR changes. The workflow downloads the
pinned public source, runs the frozen study, verifies `hashes.sha256`, checks the evidence marker
and pinned revision, confirms raw CSVs are absent from the output bundle, and uploads the bundle as
a workflow artifact.
