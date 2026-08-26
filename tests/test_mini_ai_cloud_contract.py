from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from gpu_scheduler_lab.integrations import CONTRACT_VERSION, RESULT_CONTRACT_VERSION


def _schema(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path("contracts", name).read_text(encoding="utf-8")))


def test_input_schema_freezes_v1_and_allows_forward_compatible_fields() -> None:
    schema = _schema("mini-ai-cloud-v1.schema.json")

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["contract_version"]["const"] == CONTRACT_VERSION
    assert schema["additionalProperties"] is True
    worker = schema["properties"]["workers"]["items"]
    task = schema["properties"]["tasks"]["items"]
    assert worker["additionalProperties"] is True
    assert task["additionalProperties"] is True


def test_result_schema_requires_simulated_evidence_kind() -> None:
    schema = _schema("result-handoff-v1.schema.json")

    assert schema["properties"]["contract_version"]["const"] == RESULT_CONTRACT_VERSION
    assert schema["properties"]["evidence_kind"]["const"] == "SIMULATED"
    assert set(schema["required"]) == {
        "contract_version",
        "evidence_kind",
        "limitations",
        "results",
    }
