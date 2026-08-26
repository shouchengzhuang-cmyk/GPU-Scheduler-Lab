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
    "jains_fairness_index",
    "sla_violation_rate",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_study(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0.0":
        raise ValueError("public trace study config must use schema_version 1.0.0")
    study = raw.get("public_trace_study")
    if not isinstance(study, dict):
        raise ValueError("config must contain public_trace_study")
    if study.get("evidence_kind") != EVIDENCE_KIND:
        raise ValueError(f"evidence_kind must be {EVIDENCE_KIND}")
    return study


def _verify_source(input_dir: Path, study: dict[str, Any]) -> dict[str, Any]:
    manifest_path = input_dir / SOURCE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {manifest_path}; use scripts/download_trace.py so hashes are recorded"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("hash_algorithm") != "sha256":
        raise ValueError("source manifest must contain SHA-256 hashes")

    dataset = study.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("study dataset config must be a mapping")
    version = str(dataset.get("version", ""))
    source_ref = str(dataset.get("source_ref", ""))
    if manifest.get("dataset_version") != version:
        raise ValueError("source manifest dataset version does not match frozen study config")
    if manifest.get("source_ref") != source_ref:
        raise ValueError("source manifest revision does not match frozen study config")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("source manifest files must be a list")
    by_name = {
        str(item.get("file")): item
        for item in files
        if isinstance(item, dict) and item.get("file")
    }
    required_files = dataset.get("required_files")
    if not isinstance(required_files, list):
        raise ValueError("dataset.required_files must be a list")
    for required in map(str, required_files):
        if required == SOURCE_MANIFEST:
            continue
        if required not in by_name:
            raise ValueError(f"source manifest does not identify required file {required}")

    for name, item in sorted(by_name.items()):
        if Path(name).name != name:
            raise ValueError(f"unsafe source filename in manifest: {name!r}")
        expected = item.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"source manifest has no valid SHA-256 for {name}")
        source_path = input_dir / name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        actual = _sha256_file(source_path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
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


def _require_columns(
    reader: csv.DictReader[str], required: set[str], path: Path
) -> list[str]:
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
                counts = stats["eligible_models"]
                counts[model] = int(counts.get(model, 0)) + 1
    if not stats["eligible_rows"]:
        raise ValueError("no eligible nodes remain after normalization")
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
                    gpu_request = _positive_integer(row, "gpu_request")
                    _positive_integer(row, "worker_num")
                    submit_time = _finite(row, "submit_time")
                    duration = _finite(row, "duration")
                    cpu_request = _finite(row, "cpu_request")
                    job_type = _required(row, "job_type").lower()
                    if submit_time < 0 or duration <= 0 or cpu_request < 0:
                        raise ValueError("invalid time, duration, or CPU request")
                    if job_type not in {"hp", "high-priority", "spot", "low-priority"}:
                        raise ValueError("unsupported job_type")
                except (TypeError, ValueError) as exc:
                    if "gpu_request is not a positive integer" in str(exc):
                        stats["excluded_fractional_gpu_rows"] += 1
                    else:
                        stats["excluded_invalid_rows"] += 1
                    continue
                if job_id in seen_ids:
                    stats["excluded_duplicate_id_rows"] += 1
                    continue
                seen_ids.add(job_id)
                if model not in models:
                    stats["excluded_model_rows"] += 1
                    continue
                if gpu_request <= 0:
                    stats["excluded_invalid_rows"] += 1
                    continue
                writer.writerow(row)
                submit_times.append(submit_time)
                organizations.add(organization)
                stats["eligible_rows"] += 1
                counts = stats["eligible_models"]
                counts[model] = int(counts.get(model, 0)) + 1
    if not submit_times:
        raise ValueError("no eligible jobs remain after normalization")
    stats["eligible_organizations"] = len(organizations)
    stats["eligible_ratio"] = stats["eligible_rows"] / stats["source_rows"]
    return stats, submit_times


def _window_starts(submit_times: list[float], quantiles: list[float]) -> list[float]:
    unique = sorted(set(submit_times))
    starts: list[float] = []
    for quantile in quantiles:
        if not 0 <= quantile <= 1:
            raise ValueError("window quantiles must be in [0, 1]")
        value = unique[int(quantile * (len(unique) - 1))]
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
        raise ValueError("replay window contains no source organization identities")
    share = float(scenario.cluster.total_gpu_count) / len(organizations)
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
            raise ValueError(f"job {job.id} is missing organization")
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


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["window_id", "start", "nodes", "jobs", "tenants", "policy", *SUMMARY_METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _write_report(
    path: Path,
    study: dict[str, Any],
    manifest: dict[str, Any],
    normalization: dict[str, Any],
    windows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    source_lines = []
    for item in manifest.get("files", []):
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
            "| {window_id} | {policy} | {jobs} | {tenants} | {util} | {p95} | "
            "{completion} | {fairness} |".format(
                window_id=row["window_id"],
                policy=row["policy"],
                jobs=row["jobs"],
                tenants=row["tenants"],
                util=_metric(row["average_gpu_utilization"]),
                p95=_metric(row["p95_waiting_time"]),
                completion=_metric(row["completion_rate"]),
                fairness=_metric(row["jains_fairness_index"]),
            )
        )
    window_lines = [
        f"- `{item['id']}`: start={item['start']}, duration={item['duration']}, "
        f"nodes={item['nodes']}, jobs={item['jobs']}, tenants={item['tenants']}"
        for item in windows
    ]
    claims = "\n".join(f"- {claim}" for claim in study.get("claims", []))
    nodes = normalization["nodes"]
    jobs = normalization["jobs"]
    dataset = study["dataset"]
    report = f"""# Public trace study report

**Study:** `{study['id']}`  
**Evidence kind:** `{EVIDENCE_KIND}`  
**Source:** `{dataset['version']}@{dataset['source_ref']}`

## Evidence boundary

This is deterministic **discrete-event replay evidence**. It is not a real-GPU, Kubernetes,
vLLM, Alibaba production, or production-scheduler measurement.

{claims}

## Source identity and hashes

Pinned source: {dataset['source_url']}

{chr(10).join(source_lines)}

The raw README and CSV files are verified before preprocessing but are not copied into this
result bundle and are not committed to Git.

## Normalization coverage

- Node rows: {nodes['source_rows']} source; {nodes['eligible_rows']} eligible;
  {nodes['excluded_model_rows']} model exclusions; {nodes['excluded_invalid_rows']} invalid;
  {nodes['excluded_duplicate_id_rows']} duplicate IDs.
- Job rows: {jobs['source_rows']} source; {jobs['eligible_rows']} eligible
  ({jobs['eligible_ratio']:.2%}); {jobs['excluded_model_rows']} model exclusions;
  {jobs['excluded_fractional_gpu_rows']} fractional/non-integer GPU exclusions;
  {jobs['excluded_invalid_rows']} other invalid rows;
  {jobs['excluded_duplicate_id_rows']} duplicate IDs.
- Eligible source organizations: {jobs['eligible_organizations']}.
- Frozen GPU-memory map: `{normalization['gpu_memory_gb']}`.

Source facts remain source facts: IDs, organization, GPU model, CPU/GPU request, worker count,
submit time, duration, HP/Spot class, and node capacity. Arrival normalization, aggregate integer
GPU count, simulator priority, mapped GPU memory, queue IDs, weights, and guarantees are derived
or synthetic and are labeled as such.

## Synthetic tenant overlay

`organization` supplies only tenant identity. Queue paths, weights, and equal GPU-unit guarantees
are synthetic controls. Borrowing and reclaim are enabled to compare the fair-share policies on
the same source-derived workload; none of these values are Alibaba production configuration.

## Replay windows

{chr(10).join(window_lines)}

Window anchors are deterministic quantiles of distinct eligible submit times. Each window is
bounded by the frozen node/job limits in `study-config.snapshot.yaml`.

## Results

{chr(10).join(result_lines)}

Machine-readable metrics are in `results.json` and `summary.csv`.

## Reproduce

```bash
python scripts/download_trace.py --output-dir .data/alibaba-spot-gpu-v2026
python scripts/run_public_trace_study.py \\
  --input .data/alibaba-spot-gpu-v2026 \\
  --config study/public-trace-study.yaml \\
  --output-dir build/public-trace-study
cd build/public-trace-study && sha256sum -c hashes.sha256
```

## Limitations

- The source has no rack/zone topology, so none is invented.
- Fractional GPU requests are outside the current integer-GPU simulator and are excluded.
- A bounded replay is evidence about this simulator under source-derived workloads, not a claim
  about Alibaba production behavior or production scheduler superiority.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def _write_hashes(output_dir: Path) -> None:
    lines = [
        f"{_sha256_file(path)}  {path.name}"
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "hashes.sha256"
    ]
    (output_dir / "hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_dir: Path, config_path: Path, output_dir: Path) -> None:
    study = _load_study(config_path)
    manifest = _verify_source(input_dir, study)
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
            "dataset_version": study["dataset"]["version"],
            "source_ref": study["dataset"]["source_ref"],
            "nodes": node_stats,
            "jobs": job_stats,
            "gpu_memory_gb": gpu_memory,
            "synthetic_fields": study["tenant_overlay"]["synthetic_fields"],
        }
        _write_json(output_dir / "normalization-report.json", normalization)

        result_windows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        window_summary: list[dict[str, Any]] = []
        starts = _window_starts(submit_times, quantiles)
        for index, start in enumerate(starts):
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
            if not scenario.jobs:
                raise ValueError(f"replay window starting at {start} selected no jobs")
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
            window_summary.append(
                {
                    key: window_metadata[key]
                    for key in ("id", "start", "duration", "nodes", "jobs", "tenants")
                }
            )
            policy_results: list[dict[str, Any]] = []
            for policy in policies:
                run_scenario = scenario.clone()
                scheduler = create_scheduler(policy, run_scenario)
                result = Simulator.from_scenario(run_scenario, scheduler).run()
                policy_results.append({"policy": policy, "metrics": result.metrics})
                summary_rows.append(
                    {
                        "window_id": window_id,
                        "start": start,
                        "nodes": len(scenario.cluster.nodes),
                        "jobs": len(scenario.jobs),
                        "tenants": overlay["tenant_count"],
                        "policy": policy,
                        **{name: result.metrics[name] for name in SUMMARY_METRICS},
                    }
                )
            result_windows.append({**window_metadata, "policies": policy_results})

    payload = {
        "schema_version": "1.0.0",
        "study_id": study["id"],
        "evidence_kind": EVIDENCE_KIND,
        "dataset_version": study["dataset"]["version"],
        "source_ref": study["dataset"]["source_ref"],
        "source_manifest": SOURCE_MANIFEST,
        "config_snapshot": "study-config.snapshot.yaml",
        "windows": result_windows,
        "limitations": [
            "Discrete-event simulation; not a real-GPU, Kubernetes, vLLM, or production result.",
            "Tenant queue guarantees are synthetic experiment controls.",
            "Replay uses deterministic bounded windows after full-source hashing and validation.",
        ],
    }
    _write_json(output_dir / "results.json", payload)
    _write_summary(output_dir / "summary.csv", summary_rows)
    _write_json(output_dir / "windows.json", window_summary)
    _write_report(
        output_dir / "report.md",
        study,
        manifest,
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
