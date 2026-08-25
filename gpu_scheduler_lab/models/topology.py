from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class TopologyMode(StrEnum):
    NONE = "none"
    PREFER_SAME_NODE = "prefer_same_node"
    PREFER_SAME_RACK = "prefer_same_rack"
    REQUIRE_SAME_NODE = "require_same_node"
    REQUIRE_SAME_RACK = "require_same_rack"


def topology_domain(node_id: str, topology: dict[str, str], level: str) -> str:
    if level == "node":
        return node_id
    value = topology.get(level)
    if value:
        return value
    return f"__{level}_unknown__:{node_id}"


def topology_distance(
    node_a_id: str,
    topology_a: dict[str, str],
    node_b_id: str,
    topology_b: dict[str, str],
) -> int:
    if node_a_id == node_b_id:
        return 0
    rack_a = topology_domain(node_a_id, topology_a, "rack")
    rack_b = topology_domain(node_b_id, topology_b, "rack")
    if rack_a == rack_b:
        return 1
    zone_a = topology_domain(node_a_id, topology_a, "zone")
    zone_b = topology_domain(node_b_id, topology_b, "zone")
    if zone_a == zone_b:
        return 2
    return 3


def topology_requirement_satisfied(
    mode: TopologyMode, node_ids: Iterable[str], topologies: dict[str, dict[str, str]]
) -> bool:
    unique = sorted(set(node_ids))
    if len(unique) <= 1 or mode in {
        TopologyMode.NONE,
        TopologyMode.PREFER_SAME_NODE,
        TopologyMode.PREFER_SAME_RACK,
    }:
        return True
    if mode is TopologyMode.REQUIRE_SAME_NODE:
        return len(unique) == 1
    if mode is TopologyMode.REQUIRE_SAME_RACK:
        racks = {topology_domain(node_id, topologies[node_id], "rack") for node_id in unique}
        return len(racks) == 1
    return True
