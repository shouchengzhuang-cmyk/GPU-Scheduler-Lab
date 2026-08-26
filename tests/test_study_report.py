from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from gpu_scheduler_lab.study import StudyConfig
from gpu_scheduler_lab.study.report import generate_study_report, verify_hash_manifest
from gpu_scheduler_lab.study.runner import ScenarioTemplate, StudyRunPlan, run_study

ROOT = Path(__file__).parents[1]


def test_report_bundle_is_summary_backed_hashed_and_tamper_evident(tmp_path: Path) -> None:
    config_path = _temporary_overlay(tmp_path)

    def executor(
        _config: StudyConfig,
        _template: ScenarioTemplate,
        plan: StudyRunPlan,
    ) -> dict[str, float]:
        return {
            "completion-rate": 0.5 + (int(plan.run_id[:2], 16) % 5) / 10,
            "average-gpu-utilization": 0.4,
            "p95-wait": 3.0,
            "jain-service-quality-fairness": 0.8,
        }

    study = run_study(config_path, executor=executor)
    report = generate_study_report(study.output_directory)

    assert report.report.is_file()
    assert all(path.stat().st_size > 0 for path in report.tables + report.figures)
    text = report.report.read_text(encoding="utf-8")
    for heading in (
        "## Research question",
        "## Model assumptions",
        "## Method",
        "## Results",
        "## Sensitivity",
        "## Ablation",
        "## Limitations",
        "## Reproducibility instructions",
    ):
        assert heading in text
    first_mean = json.loads(study.summary_json.read_text(encoding="utf-8"))["summary"][0]["mean"]
    assert f"{float(first_mean):.6g}" in text
    verified = verify_hash_manifest(study.output_directory)
    assert verified > 10
    hashes = report.hashes.read_text(encoding="utf-8")
    assert "  report.md" in hashes
    assert "  figures/baseline-key-metrics.png" in hashes

    report.report.write_text(text + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_hash_manifest(study.output_directory)


def test_hash_manifest_rejects_unlisted_artifact(tmp_path: Path) -> None:
    config_path = _temporary_overlay(tmp_path)

    def executor(
        _config: StudyConfig,
        _template: ScenarioTemplate,
        _plan: StudyRunPlan,
    ) -> dict[str, float]:
        return {"completion-rate": 1.0}

    study = run_study(config_path, executor=executor)
    generate_study_report(study.output_directory)
    (study.output_directory / "unlisted.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(ValueError, match="coverage mismatch"):
        verify_hash_manifest(study.output_directory)


def _temporary_overlay(tmp_path: Path) -> Path:
    for document in ("schema.json", "hypotheses.md", "metric-definitions.md"):
        shutil.copyfile(ROOT / "study" / document, tmp_path / document)
    overlay = {
        "extends": str((ROOT / "study" / "study.yaml").resolve()),
        "scenario": str((ROOT / "study" / "scenarios" / "small.yaml").resolve()),
        "execution": {
            "seeds": [5],
            "output_directory": str(tmp_path / "output"),
            "grid_mode": "one-at-a-time",
            "warmup_runs": 0,
            "replications_per_seed": 1,
            "max_retries": 0,
            "resume": True,
            "ablations": ["topology"],
        },
    }
    path = tmp_path / "study-overlay.yaml"
    path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    return path
