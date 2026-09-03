from __future__ import annotations

from enum import StrEnum


class AcceleratorVendor(StrEnum):
    UNKNOWN = "unknown"
    NVIDIA = "nvidia"
    HUAWEI_ASCEND = "huawei-ascend"


class AcceleratorKind(StrEnum):
    GPU = "gpu"
    NPU = "npu"


class AcceleratorSelectionPolicy(StrEnum):
    ANY = "any"


def vendor_supports_kind(vendor: AcceleratorVendor, kind: AcceleratorKind) -> bool:
    return (vendor, kind) in {
        (AcceleratorVendor.NVIDIA, AcceleratorKind.GPU),
        (AcceleratorVendor.HUAWEI_ASCEND, AcceleratorKind.NPU),
    }
