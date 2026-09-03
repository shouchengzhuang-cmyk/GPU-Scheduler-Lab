from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

from gpu_scheduler_lab import __version__


def test_package_cli_and_runtime_versions_are_consistent() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert project["name"] == "gpu-scheduler-lab"
    assert project["version"] == __version__ == version("gpu-scheduler-lab") == "0.4.0"
    assert project["scripts"] == {"gpu-scheduler-lab": "gpu_scheduler_lab.cli:main"}
