from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from gpu_scheduler_lab.scenario import Scenario, scenario_to_dict


def scenario_hash(scenario: Scenario) -> str:
    canonical = json.dumps(
        scenario_to_dict(scenario),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def git_sha(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def python_version() -> str:
    return sys.version.split()[0]
