from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_scheduler_lab.cli import main
from gpu_scheduler_lab.study import (
    FORMAL_METRIC_IDS,
    FORMAL_POLICY_IDS,
    FORMAL_VARIABLE_IDS,
    StudyConfig,
)

ROOT = Path(__file__).parents[1]
STUDY_CONFIG = ROOT / "study" / "study.yaml"


def test_canonical_study_freezes_policies_metrics_variables_and_hypotheses() -> None:
    config = StudyConfig.load(STUDY_CONFIG)

    assert tuple(policy.id for policy in config.policies) == FORMAL_POLICY_IDS
    assert tuple(metric.id for metric in config.metrics) == FORMAL_METRIC_IDS
    assert tuple(variable.id for variable in config.variables) == FORMAL_VARIABLE_IDS
    assert config.hypotheses
    assert all(hypothesis.independent_variable_ids for hypothesis in config.hypotheses)
    assert all(hypothesis.dependent_metric_ids for hypothesis in config.hypotheses)
    assert config.scenario_path.is_file()
    assert config.output_directory == (ROOT / "build" / "study" / "canonical").resolve()
    assert json.loads((ROOT / "study" / "schema.json").read_text(encoding="utf-8"))


def test_study_validate_cli_reports_contract(capsys: pytest.CaptureFixture[str]) -> None:
    main(["study", "validate", "--config", str(STUDY_CONFIG)])

    output = capsys.readouterr().out
    assert "4 policies" in output
    assert "13 metrics" in output
    assert "5 variables" in output
    assert "5 hypotheses" in output


def test_study_validate_cli_rejects_unregistered_policy() -> None:
    with pytest.raises(ValueError, match="not registered"):
        main(
            [
                "study",
                "validate",
                "--config",
                str(STUDY_CONFIG),
                "--policy",
                "fifo",
            ]
        )


def test_study_config_rejects_duplicate_policy(tmp_path: Path) -> None:
    config_path = _mutated_config(
        tmp_path,
        lambda payload: payload["policies"].append("binpack"),
    )

    with pytest.raises(ValueError, match="duplicate formal policy ids"):
        StudyConfig.load(config_path)


def test_study_config_rejects_unknown_hypothesis_metric(tmp_path: Path) -> None:
    def add_unknown_metric(payload: dict[str, Any]) -> None:
        payload["hypotheses"][0]["dependent_metric_ids"].append("unknown-metric")

    config_path = _mutated_config(tmp_path, add_unknown_metric)

    with pytest.raises(ValueError, match="dependent metric ids are unknown"):
        StudyConfig.load(config_path)


def _mutated_config(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    payload = yaml.safe_load(STUDY_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["policy_registry"] = str((ROOT / "study" / "policies.yaml").resolve())
    payload["scenario"] = str((ROOT / "study" / "scenarios" / "canonical.yaml").resolve())
    mutate(payload)
    for document in ("schema.json", "hypotheses.md", "metric-definitions.md"):
        shutil.copyfile(ROOT / "study" / document, tmp_path / document)
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path
