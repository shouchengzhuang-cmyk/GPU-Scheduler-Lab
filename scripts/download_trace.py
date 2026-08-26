from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2026-spot-gpu"
)
FILES = ("node_info_df.csv", "job_info_df.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Alibaba 2026 spot-GPU trace")
    parser.add_argument("--output-dir", type=Path, default=Path(".data/alibaba-spot-gpu-v2026"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, str | int]] = []
    for name in FILES:
        destination = args.output_dir / name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite")
        source = f"{BASE_URL}/{name}"
        with urllib.request.urlopen(source) as response:  # noqa: S310
            payload = response.read()
        destination.write_bytes(payload)
        downloaded.append({"file": name, "source_url": source, "bytes": len(payload)})
    manifest = {
        "dataset": "Alibaba cluster-trace-v2026-spot-gpu",
        "source": "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-spot-gpu",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "files": downloaded,
    }
    with (args.output_dir / "source-manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Downloaded trace to {args.output_dir}")


if __name__ == "__main__":
    main()
