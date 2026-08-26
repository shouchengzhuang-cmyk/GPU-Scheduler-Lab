from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from gpu_scheduler_lab.queues.model import QueueSpec, ResourceVector
from gpu_scheduler_lab.schedulers import create_scheduler
from gpu_scheduler_lab.simulator.engine import Simulator
from gpu_scheduler_lab.traces import AlibabaSpotGPUTraceAdapter, TraceFilter

NODE_FILE = "node_info_df.csv"
JOB_FILE = "job_info_df.csv"
SOURCE_MANIFEST = "source-manifest.json"
EVIDENCE_KIND = "SIMULATED_TRACE_REPLAY"
SUMMARY_METRICS = (
    "average_gpu_utilization",
    "average_waiting_time",
    "p95_waiting_time",
    "completion_rate",
    "gpu_count_fragmentation",
    "gpu_memory_fragmentation",
    "queue_service_jains_index",
    "sla_violation_rate",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0.0":
        raise ValueError("public trace study config must use schema_version 1.0.0")
    study = raw.get("public_trace_study")
    if not isinstance(study, dict):
        raise ValueError("config must contain public_trace_study")
    if study.get("evidence_kind") != EVIDENCE_KIND:
        raise ValueError(f"evidence_kind must be {EVIDENCE_KIND}")
    return raw, study


def _verify_source(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / SOURCE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {manifest_path}; use scripts/download_trace.py so input hashes are recorded"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("hash_algorithm") != "sha256":
        raise ValueError("source manifest must contain SHA-256 hashes")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("source manifest files must be a list")
    by_name = {
        str(item.get("file")): item for item in files if isinstance(item, dict) and item.get("file")
    }
    for name in (NODE_FILE, JOB_FILE):
        source_path = input_dir / name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        item = by_name.get(name)
        if item is None or not isinstance(item.get("sha256"), str):
            raise ValueError(f"source manifest has no SHA-256 for {name}")
        actual = _sha256_file(source_path)
        if actual != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {item['sha256']}")
    return manifest


def _required(row: dict[str, str], key: str) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is empty")
    return value


def _finite(row: dict[str, str], key: str) -> float:
    value = float(_required(row, key))
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def _positive_integer(row: dict[str, str], key: str) -> int:
    value = _finite(row, key)
    if value <= 0 or not value.is_integer():
        raise ValueError(f"{key} is not a positive integer")
    return int(value)


def _require_columns(reader: csv.DictReader[str], required: set[str], path: Path) -> list[str]:
    fieldnames = list(reader.fieldnames or [])
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    return fieldnames


def _filter_nodes(source: Path, destination: Path, models: set[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "source_rows": 0,
        "eligible_rows": 0,
        "excluded_model_rows": 0,
        "excluded_invalid_rows": 0,
        "excluded_duplicate_id_rows": 0,
        "eligible_models": {},
    }
    seen_ids: set[str] = set()
    with source.open(encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = _require_columns(
            reader,
            {"node_name", "gpu_model", "gpu_capacity_num", "cpu_num"},
            source,
        )
        with destination.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                stats["source_rows"] += 1
                try:
                    node_id = _required(row, "node_name")
                    model = _required(row, "gpu_model")
                    _positive_integer(row, "gpu_capacity_num")
                    if _finite(row, "cpu_num") <= 0:
                        raise ValueError("cpu_num is not positive")
                except (TypeError, ValueError):
                    stats["excluded_invalid_rows"] += 1
                    continue
                if node_id in seen_ids:
                    stats["excluded_duplicate_id_rows"] += 1
                    continue
                seen_ids.add(node_id)
                if model not in models:
                    stats["excluded_model_rows"] += 1
                    continue
                writer.writerow(row)
                stats["eligible_rows"] += 1
                model_counts = stats["eligible_models"]
                model_counts[model] = int(model_counts.get(model, 0)) + 1
    if stats["eligible_rows"] == 0:
        raise ValueError("no eligible nodes remain after public-trace normalization")
    return stats


def _filter_jobs(
    source: Path,
    destination: Path,
    models: set[str],
) -> tuple[dict[str, Any], list[float]]:
    stats: dict[str, Any] = {
        "source_rows": 0,
        "eligible_rows": 0,
        "excluded_model_rows": 0,
        "excluded_fractional_gpu_rows": 0,
        "excluded_invalid_rows": 0,
        "excluded_duplicate_id_rows": 0,
        "eligible_models": {},
        "eligible_organizations": 0,
    }
    submit_times: list[float] = []
    organizations: set[str] = set()
    seen_ids: set[str] = set()
    required = {
        "job_name",
        "organization",
        "gpu_model",
        "cpu_request",
        "gpu_request",
        "worker_num",
        "submit_time",
        "duration",
        "job_type",
    }
    with source.open(encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = _require_columns(reader, required, source)
        with destination.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                stats["source_rows"] += 1
                try:
                    job_id = _required(row, "job_name")
                    organization = _required(row, "organization")
                    model = _required(row, "gpu_model")
                    gpu_request = _finite(row, "gpu_request")
                    worker_num = _finite(row, "worker_num")
                    submit_time = _finite(row, "submit_time")
                    duration = _finite(row, "duration")
                    cpu_request = _finite(row, "cpu_request")
                    job_type = _required(row, "job_type").lower()
                    if submit_time < 0 or duration <= 0 or cpu_request < 0:
                        raise ValueError("invalid time, duration, or CPU request")
                    if worker_num <= 0 or not worker_num.is_integer():
                        raise ValueError("worker_num is not a positive integer")
                    if job_type not in {"hp", "high-priority", "spot", "low-priority"}:
                        raise ValueError("unsupported job_type")
                except (TypeError, ValueError):
                    stats["excluded_invalid_rows"] += 1
                    continue
                if job_id in seen_ids:
                    stats["excluded_duplicate_id_rows"] += 1
                    continue
                seen_ids.add(job_id)
                if model not in models:
                    stats["excluded_model_rows"] += 1
                    continue
                if gpu_request <= 0 or not gpu_request.is_integer():
                    stats["excluded_fractional_gpu_rows"] += 1
                    continue
                writer.writerow(row)
                submit_times.append(submit_time)
                organizations.add(organization)
                stats["eligible_rows"] += 1
                model_counts = stats["eligible_models"]
                model_counts[model] = int(model_counts.get(model, 0)) + 1
    if not submit_times:
        raise ValueError("no eligible jobs remain after public-trace normalization")
    stats["eligible_organizations"] = len(organizations)
    stats["eligible_ratio"] = stats["eligible_rows"] / stats["source_rows"]
    return stats, submit_times


def _window_starts(submit_times: list[float], quantiles: list[float]) -> list[float]:
    unique = sorted(set(submit_times))
    starts: list[float] = []
    for quantile in quantiles:
        if not 0 <= quantile <= 1:
            raise ValueError("window quantiles must be in [0, 1]")
        index = int(quantile * (len(unique) - 1))
        value = unique[index]
        if value not in starts:
            starts.append(value)
    return starts


def _queue_id(organization: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", organization).strip("-.").lower()
    slug = slug[:32] or "tenant"
    suffix = hashlib.sha256(organization.encode()).hexdigest()[:8]
    return f"root/org-{slug}-{suffix}"


def _apply_tenant_overlay(scenario: Any) -> dict[str, Any]:
    organizations = sorted({str(job.group) for job in scenario.jobs if job.group is not None})
    if not organizations:
        raise ValueError("selected replay window contains no source organization identities")
    total_gpu_units = float(scenario.cluster.total_gpu_count)
    share = total_gpu_units / len(organizations)
    mapping = {organization: _queue_id(organization) for organization in organizations}
    scenario.queues = tuple(
        QueueSpec(
            id=mapping[organization],
            parent="root",
            weight=1.0,
            guaranteed=ResourceVector(gpu_units=share),
            borrowing_enabled=True,
            reclaimable=True,
        )
        for organization in organizations
    )
    for job in scenario.jobs:
        if job.group is None:
            raise ValueError(f"job {job.id} is missing organization after normalization")
        job.queue_id = mapping[str(job.group)]
    overlay = {
        "kind": "synthetic_equal_share_queue_overlay",
        "source_identity_field": "organization",
        "tenant_count": len(organizations),
        "guaranteed_gpu_units_per_tenant": share,
        "borrowing_enabled": True,
        "reclaimable": True,
        "synthetic_fields": ["queue_id", "queue_weight", "guaranteed_gpu_units"],
    }
    scenario.metadata["synthetic_tenant_overlay"] = overlay
    return overlay


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "n/a"
    return str(value)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "window_id",
        "start",
        "nodes",
        "jobs",
        "tenants",
        "policy",
        "elapsed_seconds",
        *SUMMARY_METRICS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    study: dict[str, Any],
    source_manifest: dict[str, Any],
    normalization: dict[str, Any],
    windows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    source_lines = []
    for item in source_manifest.get("files", []):
        if isinstance(item, dict):
            source_lines.append(
                f"- `{item.get('file')}` — SHA-256 `{item.get('sha256')}`, "
                f"{item.get('bytes')} bytes"
            )
    result_lines = [
        "| Window | Policy | Jobs | Tenants | GPU util | P95 wait | Completion | Jain fairness |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        result_lines.append(
            "| {window_id} | {policy} | {jobs} | {tenants} | {util} | {p95} | {completion} | "
            "{fairness} |".format(
                window_id=row["window_id"],
                policy=row["policy"],
                jobs=row["jobs"],
                tenants=row["tenants"],
                util=_format_metric(row.get("average_gpu_utilization")),
                p95=_format_metric(row.get("p95_waiting_time")),
                completion=_format_metric(row.get("completion_rate")),
                fairness=_format_metric(row.get("queue_service_jains_index")),
            )
        )
    window_lines = [
        f"- `{item['id']}`: start={item['start']}, duration={item['duration']}, "
        f"nodes={item['nodes']}, jobs={item['jobs']}, tenants={item['tenants']}"
        for item in windows
    ]
    node_stats = normalization["nodes"]
    job_stats = normalization["jobs"]
    claims = "\n".join(f"- {claim}" for claim in study.get("claims", []))
    report = f"""# Public trace study report

**Study:** `{study['id']}`  
**Evidence kind:** `{EVIDENCE_KIND}`  
**Source dataset:** `{study['dataset']['version']}`

## Evidence boundary

This bundle is a deterministic **discrete-event replay study**. It is not a real-GPU,
Kubernetes, vLLM, or production scheduling measurement. The full public source files are
downloaded and SHA-256 verified before preprocessing; the simulator then executes bounded,
deterministic windows so the experiment remains reproducible and reviewable.

{claims}

## Source identity and hashes

Source: {study['dataset']['source_url']}

{chr(10).join(source_lines)}

The raw CSV files are intentionally not copied into this result bundle or committed to Git.
`source-manifest.json` is sufficient to identify the exact downloaded inputs.

## Normalization coverage

- Node rows: {node_stats['source_rows']} source, {node_stats['eligible_rows']} eligible,
  {node_stats['excluded_model_rows']} excluded for model mapping,
  {node_stats['excluded_invalid_rows']} invalid, and
  {node_stats['excluded_duplicate_id_rows']} duplicate IDs.
- Job rows: {job_stats['source_rows']} source, {job_stats['eligible_rows']} eligible
  ({job_stats['eligible_ratio']:.2%}), {job_stats['excluded_model_rows']} excluded for model
  mapping, {job_stats['excluded_fractional_gpu_rows']} fractional-GPU rows excluded,
  {job_stats['excluded_invalid_rows']} invalid, and
  {job_stats['excluded_duplicate_id_rows']} duplicate IDs.
- Eligible source organizations: {job_stats['eligible_organizations']}.
- GPU memory mappings used: `{study['normalization']['gpu_memory_gb']}`.

Original source fields remain source facts: job ID, organization, GPU model, CPU/GPU request,
worker count, submit time, duration, HP/Spot class, node ID, GPU model/capacity and CPU capacity.
Derived fields include normalized arrival time, integer aggregate GPU count, simulator priority,
and mapped GPU memory. Rows needing unsupported fractional GPU allocation or an unaudited memory
mapping are excluded rather than guessed.

## Synthetic tenant overlay

`organization` is used only as the source tenant identity. The queue path, queue weight and
equal GPU-unit guarantee are **synthetic experimental controls**. They do not exist in the
Alibaba trace and must not be presented as production policy. Borrowing and reclaim are enabled
so the two fair-share policies can be compared on the same replay workload.

## Replay windows

{chr(10).join(window_lines)}

Window anchors are deterministic quantiles over distinct eligible source submit times. Each
window is capped by the frozen `max_nodes_per_window` and `max_jobs_per_window` values in
`study-config.snapshot.yaml`.

## Results

{chr(10).join(result_lines)}

The machine-readable metrics are in `results.json` and `summary.csv`. Wall-clock simulator
runtime is reported only as local execution cost; it is not GPU latency or serving latency.

## Reproduction

```bash
python scripts/download_trace.py --output-dir .data/alibaba-spot-gpu-v2026
python scripts/run_public_trace_study.py \\
  --input .data/alibaba-spot-gpu-v2026 \\
  --config study/public-trace-study.yaml \\
  --output-dir build/public-trace-study
cd build/public-trace-study && sha256sum -c hashes.sha256
```

## Limitations

- The trace has no rack/zone topology, so this study does not invent topology.
- Fractional GPU requests are outside the current simulator resource model and are explicitly
  counted as exclusions.
- Models without a frozen memory-capacity mapping are excluded rather than inferred silently.
- A bounded replay is evidence about this simulator under these source-derived workloads, not a
  claim about Alibaba production behavior or production scheduler superiority.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def _write_hashes(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "hashes.sha256":
            lines.append(f"{_sha256_file(path)}  {path.name}")
    (output_dir / "hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_dir: Path, config_path: Path, output_dir: Path) -> None:
    raw_config, study = _load_config(config_path)
    source_manifest = _verify_source(input_dir)
    configured_version = str(study["dataset"]["version"])
    if source_manifest.get("dataset_version") != configured_version:
        raise ValueError(
            "dataset version mismatch: "
            f"{source_manifest.get('dataset_version')!r} != {configured_version!r}"
        )
    gpu_memory_raw = study["normalization"]["gpu_memory_gb"]
    if not isinstance(gpu_memory_raw, dict) or not gpu_memory_raw:
        raise ValueError("normalization.gpu_memory_gb must be a non-empty mapping")
    gpu_memory = {str(model): float(memory) for model, memory in gpu_memory_raw.items()}
    if any(not math.isfinite(memory) or memory <= 0 for memory in gpu_memory.values()):
        raise ValueError("GPU memory mappings must be finite and positive")

    replay = study["replay"]
    policies = [str(item) for item in replay["policies"]]
    quantiles = [float(item) for item in replay["window_quantiles"]]
    duration = float(replay["window_duration_seconds"])
    max_nodes = int(replay["max_nodes_per_window"])
    max_jobs = int(replay["max_jobs_per_window"])
    seed = int(replay["seed"])
    if duration <= 0 or max_nodes <= 0 or max_jobs <= 0:
        raise ValueError("replay duration and bounds must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_dir / SOURCE_MANIFEST, output_dir / SOURCE_MANIFEST)
    shutil.copyfile(config_path, output_dir / "study-config.snapshot.yaml")

    with tempfile.TemporaryDirectory(prefix="gpu-scheduler-public-trace-") as temp:
        filtered = Path(temp)
        node_stats = _filter_nodes(input_dir / NODE_FILE, filtered / NODE_FILE, set(gpu_memory))
        job_stats, submit_times = _filter_jobs(
            input_dir / JOB_FILE,
            filtered / JOB_FILE,
            set(gpu_memory),
        )
        normalization = {
            "evidence_kind": EVIDENCE_KIND,
            "dataset_version": configured_version,
            "nodes": node_stats,
            "jobs": job_stats,
            "gpu_memory_gb": gpu_memory,
            "unsupported_source_fields": ["fractional_gpu_request"],
            "synthetic_fields": study["tenant_overlay"]["synthetic_fields"],
        }
        _write_json(output_dir / "normalization-report.json", normalization)

        result_windows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        window_summary: list[dict[str, Any]] = []
        for index, start in enumerate(_window_starts(submit_times, quantiles)):
            adapter = AlibabaSpotGPUTraceAdapter(filtered, gpu_memory_gb=gpu_memory)
            trace_filter = TraceFilter(
                start=start,
                duration=duration,
                max_jobs=max_jobs,
                max_nodes=max_nodes,
                sample_rate=1.0,
                seed=seed,
                skip_invalid=False,
            )
            scenario = adapter.to_scenario(trace_filter)
            overlay = _apply_tenant_overlay(scenario)
            window_id = f"w{index:02d}"
            window_metadata = {
                "id": window_id,
                "start": start,
                "duration": duration,
                "nodes": len(scenario.cluster.nodes),
                "jobs": len(scenario.jobs),
                "tenants": overlay["tenant_count"],
                "adapter_metadata": scenario.metadata,
            }
            window_summary.append({key: window_metadata[key] for key in ("id", "start", "duration", "nodes", "jobs", "tenants")})
            policy_results: list[dict[str, Any]] = []
            for policy in policies:
                run_scenario = scenario.clone()
                scheduler = create_scheduler(policy, run_scenario)
                result = Simulator.from_scenario(run_scenario, scheduler).run()
                policy_results.append(
                    {
                        "policy": policy,
                        "elapsed_seconds": result.elapsed_seconds,
                        "metrics": result.metrics,
                    }
                )
                summary_rows.append(
                    {
                        "window_id": window_id,
                        "start": start,
                        "nodes": len(scenario.cluster.nodes),
                        "jobs": len(scenario.jobs),
                        "tenants": overlay["tenant_count"],
                        "policy": policy,
                        "elapsed_seconds": result.elapsed_seconds,
                        **{name: result.metrics.get(name) for name in SUMMARY_METRICS},
                    }
                )
            result_windows.append({**window_metadata, "policies": policy_results})

    payload = {
        "schema_version": "1.0.0",
        "study_id": study["id"],
        "evidence_kind": EVIDENCE_KIND,
        "dataset_version": configured_version,
        "source_manifest": SOURCE_MANIFEST,
        "config_snapshot": "study-config.snapshot.yaml",
        "windows": result_windows,
        "limitations": [
            "Discrete-event simulation; not a real-GPU, Kubernetes, vLLM, or production result.",
            "Tenant queue guarantees are synthetic experiment controls.",
            "Replay uses deterministic bounded windows after full-source ingestion and hashing.",
        ],
    }
    _write_json(output_dir / "results.json", payload)
    _write_summary(output_dir / "summary.csv", summary_rows)
    _write_json(output_dir / "windows.json", window_summary)
    _write_report(
        output_dir / "report.md",
        study,
        source_manifest,
        normalization,
        window_summary,
        summary_rows,
    )
    _write_hashes(output_dir)
    print(f"Public trace study: {output_dir / 'report.md'}")
    print(f"Evidence kind: {EVIDENCE_KIND}")
    print(f"Replay windows: {len(result_windows)}; policy runs: {len(summary_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen full-source bounded-window public trace study"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("study/public-trace-study.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/public-trace-study"))
    args = parser.parse_args()
    run(args.input, args.config, args.output_dir)


if __name__ == "__main__":
    main()
