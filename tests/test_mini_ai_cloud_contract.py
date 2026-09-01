from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from gpu_scheduler_lab.integrations import (
    CONTRACT_V2_VERSION,
    CONTRACT_VERSION,
    RESULT_CONTRACT_VERSION,
)


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


def test_input_schema_freezes_typed_v2_and_vendor_kind_pairs() -> None:
    schema = _schema("mini-ai-cloud-v2.schema.json")

    assert schema["properties"]["contract_version"]["const"] == CONTRACT_V2_VERSION
    worker = schema["properties"]["workers"]["items"]
    device = worker["properties"]["gpu_devices"]["items"]
    task = schema["properties"]["tasks"]["items"]
    assert set(device["required"]) == {
        "vendor",
        "kind",
        "model",
        "memory_total_mb",
        "runtime_profiles",
        "capabilities",
    }
    assert len(device["oneOf"]) == 2
    assert task["properties"]["selection_policy"]["const"] == "any"
    compatible_pairs = task["allOf"][1]["anyOf"]
    assert compatible_pairs[2]["properties"] == {
        "allowed_vendors": {"contains": {"const": "nvidia"}},
        "allowed_kinds": {"contains": {"const": "gpu"}},
    }
    assert compatible_pairs[3]["properties"] == {
        "allowed_vendors": {"contains": {"const": "huawei-ascend"}},
        "allowed_kinds": {"contains": {"const": "npu"}},
    }
    naive_timestamp = {
        "type": "string",
        "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?$",
    }
    for name in ("arrival_time", "queued_at", "sla_deadline"):
        assert task["properties"][name]["oneOf"][2] == naive_timestamp
    assert device["additionalProperties"] is True
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
