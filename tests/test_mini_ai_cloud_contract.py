from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from gpu_scheduler_lab.integrations import (
    CONTRACT_V2_VERSION,
    CONTRACT_VERSION,
    RESULT_CONTRACT_VERSION,
    import_mini_ai_cloud_export,
)


def _schema(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path("contracts", name).read_text(encoding="utf-8")))


def _v2_golden() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path("tests/fixtures/mini_ai_cloud/v2-golden.json").read_text(encoding="utf-8")),
    )


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
    timestamp = schema["$defs"]["timestamp"]
    for name in ("arrival_time", "queued_at", "sla_deadline"):
        assert task["properties"][name] == {"$ref": "#/$defs/timestamp"}
    assert timestamp["oneOf"][0] == {"type": "number"}
    assert timestamp["oneOf"][1]["type"] == "string"
    assert device["additionalProperties"] is True
    assert task["additionalProperties"] is True


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-25T00:00:00",
        "2024-02-29T23:59:59.25Z",
        "2000-02-29T23:59:59Z",
        "2026-08-25T00:00:00+08:00",
    ],
)
def test_v2_timestamp_schema_and_adapter_accept_supported_iso_values(timestamp: str) -> None:
    schema = _schema("mini-ai-cloud-v2.schema.json")
    pattern = schema["$defs"]["timestamp"]["oneOf"][1]["pattern"]
    payload = _v2_golden()
    task = payload["tasks"][0]

    for name in ("arrival_time", "queued_at", "sla_deadline"):
        task[name] = timestamp

    assert re.fullmatch(pattern, timestamp) is not None
    import_mini_ai_cloud_export(payload)


@pytest.mark.parametrize(
    "timestamp",
    ["2026-02-31T25:61:61", "2026-08-25 00:00:00", "2026-08-25T00:00:00+25:00"],
)
def test_v2_timestamp_schema_and_adapter_reject_unsupported_iso_values(timestamp: str) -> None:
    schema = _schema("mini-ai-cloud-v2.schema.json")
    pattern = schema["$defs"]["timestamp"]["oneOf"][1]["pattern"]
    payload = _v2_golden()
    payload["tasks"][0]["arrival_time"] = timestamp

    assert re.fullmatch(pattern, timestamp) is None
    with pytest.raises(ValueError, match="valid ISO-8601 timestamp"):
        import_mini_ai_cloud_export(payload)


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
