from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from gpu_scheduler_lab.study.invariants import (
    InvariantContract,
    generate_baseline,
    render_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the invariant golden baseline and show an explicit diff"
    )
    parser.add_argument("--write", action="store_true", help="accept and write the displayed diff")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    contract = InvariantContract.load(repo_root / "study/invariants.yaml")
    before = contract.baseline_path.read_text(encoding="utf-8")
    after = render_baseline(generate_baseline(contract))
    if before == after:
        print("Invariant golden baseline is current.")
        return

    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=str(contract.baseline_path),
        tofile="generated logical metrics",
        lineterm="",
    )
    print("\n".join(diff))
    if not args.write:
        raise SystemExit("baseline differs; inspect the diff, then rerun with --write to accept it")
    contract.baseline_path.write_text(after, encoding="utf-8", newline="\n")
    print(f"Updated {contract.baseline_path}")


if __name__ == "__main__":
    main()
