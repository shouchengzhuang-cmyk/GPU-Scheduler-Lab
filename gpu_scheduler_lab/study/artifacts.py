from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from gpu_scheduler_lab.study.config import StudyConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_worktree_dirty(cwd: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def capture_environment() -> dict[str, object]:
    try:
        package_version = importlib.metadata.version("gpu-scheduler-lab")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    return {
        "schema_version": "1.0.0",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "package_version": package_version,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "executable_name": Path(sys.executable).name,
    }


def study_source_hashes(config: StudyConfig) -> dict[str, object]:
    candidates = {
        "study-config": config.source_path,
        "scenario": config.scenario_path,
        "schema": config.source_path.parent / "schema.json",
        "hypotheses": config.source_path.parent / "hypotheses.md",
        "metric-definitions": config.source_path.parent / "metric-definitions.md",
    }
    files = [
        {
            "role": role,
            "path": path.name,
            "sha256": sha256_file(path),
        }
        for role, path in sorted(candidates.items())
        if path.is_file()
    ]
    return {"schema_version": "1.0.0", "files": files}
