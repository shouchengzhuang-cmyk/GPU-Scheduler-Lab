from __future__ import annotations

import pytest
from conftest import make_cluster

from gpu_scheduler_lab.models import GPU, Cluster, Job, Node, TopologyMode, topology_distance
from gpu_scheduler_lab.schedulers import (
    BinPackScheduler,
    FIFOScheduler,
    PreemptiveScheduler,
    SpreadScheduler,
    TopologyAwareScheduler,
)
from gpu_scheduler_lab.simulator import Simulator


def _topology_cluster() -> Cluster:
    return Cluster(
        [
            Node(
                "a",
                [GPU("a-0", "a", 80, model="A100")],
                topology={"zone": "z1", "rack": "r1"},
            ),
            Node(
                "b",
                [GPU("b-0", "b", 80, model="A100")],
                topology={"zone": "z1", "rack": "r1"},
            ),
            Node(
                "c",
                [GPU("c-0", "c", 80, model="H100")],
                topology={"zone": "z1", "rack": "r2"},
            ),
            Node(
                "d",
                [GPU("d-0", "d", 80, model="H100")],
                topology={"zone": "z2", "rack": "r3"},
            ),
        ]
    )


def test_gpu_model_compatibility_is_centralized() -> None:
    gpu = GPU("gpu", "node", 80, model="A100")

    assert gpu.can_host(Job("exact", 0, 1, 1, 40, gpu_model="A100"))
    assert not gpu.can_host(Job("wrong", 0, 1, 1, 40, gpu_model="H100"))
    assert gpu.can_host(Job("allowed", 0, 1, 1, 40, allowed_gpu_models=("A100", "H100")))
    assert not gpu.can_host(Job("memory", 0, 1, 1, 90, gpu_model="A100"))


def test_unschedulable_node_rejects_otherwise_compatible_gpu() -> None:
    cluster = make_cluster([[80]])
    cluster.nodes[0].schedulable = False
    job = Job("job", 0, 1, 1, 20)

    assert FIFOScheduler().place(cluster, job) is None
    with pytest.raises(ValueError, match="unavailable"):
        cluster.allocate(job, [cluster.nodes[0].gpus[0].id])


def test_heterogeneous_model_gang_is_all_or_nothing() -> None:
    cluster = _topology_cluster()
    job = Job("gang", 0, 10, 3, 40, gpu_model="A100", gang=True)

    assert TopologyAwareScheduler().place(cluster, job) is None
    assert all(gpu.free for gpu in cluster.gpus)


def test_topology_distance_hierarchy() -> None:
    cluster = _topology_cluster()
    nodes = {node.id: node for node in cluster.nodes}

    assert topology_distance("a", nodes["a"].topology, "a", nodes["a"].topology) == 0
    assert topology_distance("a", nodes["a"].topology, "b", nodes["b"].topology) == 1
    assert topology_distance("a", nodes["a"].topology, "c", nodes["c"].topology) == 2
    assert topology_distance("a", nodes["a"].topology, "d", nodes["d"].topology) == 3


def test_same_rack_label_in_different_zones_is_cross_zone() -> None:
    assert (
        topology_distance(
            "a",
            {"zone": "z1", "rack": "shared-name"},
            "b",
            {"zone": "z2", "rack": "shared-name"},
        )
        == 3
    )


def test_topology_scheduler_enforces_require_same_rack() -> None:
    cluster = _topology_cluster()
    job = Job(
        "gang",
        0,
        10,
        2,
        40,
        gpu_model="A100",
        gang=True,
        topology_mode=TopologyMode.REQUIRE_SAME_RACK,
    )

    assert TopologyAwareScheduler().place(cluster, job) == ["a-0", "b-0"]


def test_topology_scheduler_preference_has_stable_tie_break() -> None:
    cluster = _topology_cluster()
    job = Job(
        "gang",
        0,
        10,
        2,
        40,
        allowed_gpu_models=("A100", "H100"),
        gang=True,
        topology_mode=TopologyMode.PREFER_SAME_RACK,
    )

    first = TopologyAwareScheduler().place(cluster, job)
    second = TopologyAwareScheduler().place(cluster, job)

    assert first == second == ["a-0", "b-0"]


def test_non_gang_multi_gpu_placement_contributes_topology_distance() -> None:
    job = Job(
        "multi-gpu",
        0,
        10,
        2,
        40,
        gpu_model="A100",
        gang=False,
        topology_mode=TopologyMode.PREFER_SAME_RACK,
    )

    result = Simulator(_topology_cluster(), [job], TopologyAwareScheduler()).run()

    assert result.metrics["average_topology_distance"] == 1.0
    assert result.metrics["same_rack_gang_placement_count"] == 0


def test_all_existing_schedulers_reuse_required_topology_feasibility() -> None:
    cluster = _topology_cluster()
    job = Job(
        "gang",
        0,
        10,
        2,
        40,
        gpu_model="A100",
        gang=True,
        topology_mode=TopologyMode.REQUIRE_SAME_RACK,
    )

    for scheduler in (
        FIFOScheduler(),
        BinPackScheduler(),
        SpreadScheduler(),
        PreemptiveScheduler(),
    ):
        placement = scheduler.place(cluster, job)
        assert placement == ["a-0", "b-0"]
