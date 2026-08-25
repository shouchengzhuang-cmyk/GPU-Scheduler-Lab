from __future__ import annotations

from pathlib import Path

import pytest

from gpu_scheduler_lab.traces import AlibabaSpotGPUTraceAdapter, TraceFilter

FIXTURE = Path("tests/fixtures/alibaba_trace_sample")


def _adapter() -> AlibabaSpotGPUTraceAdapter:
    return AlibabaSpotGPUTraceAdapter(FIXTURE, gpu_memory_gb={"GPU-series-1": 24})


def test_alibaba_sample_parse_and_timestamp_normalization() -> None:
    scenario = _adapter().to_scenario(TraceFilter(max_jobs=3))

    assert len(scenario.cluster.nodes) == 4
    assert len(scenario.jobs) == 3
    assert scenario.jobs[0].arrival_time == 0
    assert scenario.jobs[2].arrival_time == 9589663
    assert scenario.jobs[2].gpu_model == "A100-SXM4-80GB"
    assert scenario.jobs[2].gpu_count == 16
    assert scenario.metadata["normalization"]["selected_rows"] == 3


def test_alibaba_window_filter_rebases_to_zero() -> None:
    scenario = _adapter().to_scenario(TraceFilter(start=9_000_000, duration=1_000_000, max_jobs=1))

    assert [job.arrival_time for job in scenario.jobs] == [0.0]
    assert scenario.metadata["normalization"]["time_origin"] == 9589663


def test_alibaba_sampling_is_deterministic() -> None:
    trace_filter = TraceFilter(sample_rate=0.75, seed=42)

    first = _adapter().to_scenario(trace_filter)
    second = _adapter().to_scenario(trace_filter)

    assert [job.id for job in first.jobs] == [job.id for job in second.jobs]
    assert first.metadata["normalization"] == second.metadata["normalization"]


def test_alibaba_invalid_schema_is_explicit(tmp_path: Path) -> None:
    (tmp_path / "node_info_df.csv").write_text("node_name\nnode\n", encoding="utf-8")
    (tmp_path / "job_info_df.csv").write_text("job_name\njob\n", encoding="utf-8")
    adapter = AlibabaSpotGPUTraceAdapter(tmp_path)

    with pytest.raises(ValueError, match="missing required columns"):
        adapter.to_scenario(TraceFilter())


def test_alibaba_unknown_gpu_memory_requires_mapping(tmp_path: Path) -> None:
    (tmp_path / "node_info_df.csv").write_text(
        "node_name,gpu_model,gpu_capacity_num,cpu_num\nn,Unknown,1,8\n",
        encoding="utf-8",
    )
    (tmp_path / "job_info_df.csv").write_text(
        "job_name,organization,gpu_model,cpu_request,gpu_request,worker_num,submit_time,duration,job_type\n"
        "j,o,Unknown,1,1,1,0,1,HP\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no memory mapping"):
        AlibabaSpotGPUTraceAdapter(tmp_path).to_scenario(TraceFilter())


def test_alibaba_skip_invalid_handles_truncated_csv_row(tmp_path: Path) -> None:
    (tmp_path / "node_info_df.csv").write_text(
        "node_name,gpu_model,gpu_capacity_num,cpu_num\nn,A10,1,8\n",
        encoding="utf-8",
    )
    (tmp_path / "job_info_df.csv").write_text(
        "job_name,organization,gpu_model,cpu_request,gpu_request,worker_num,submit_time,duration,job_type\n"
        "good,o,A10,1,1,1,0,1,HP\n"
        "truncated,o,A10,1,1\n",
        encoding="utf-8",
    )

    scenario = AlibabaSpotGPUTraceAdapter(tmp_path).to_scenario(TraceFilter(skip_invalid=True))

    assert [job.id for job in scenario.jobs] == ["good"]
    assert scenario.metadata["normalization"]["invalid_rows"] == 1
    assert scenario.metadata["warnings"] == ["row 3: worker_num must not be empty"]


def test_alibaba_dataset_absence_is_a_local_error_not_an_import_failure(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing Alibaba trace files"):
        AlibabaSpotGPUTraceAdapter(tmp_path)
