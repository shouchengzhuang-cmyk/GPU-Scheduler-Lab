from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DATASET = "Alibaba cluster-trace-v2026-spot-gpu"
DATASET_VERSION = "cluster-trace-v2026-spot-gpu"
SOURCE_REF = "c08f563115af39bad047353431bf745b4dee665c"
SOURCE = (
    "https://github.com/alibaba/clusterdata/tree/"
    f"{SOURCE_REF}/cluster-trace-v2026-spot-gpu"
)
BASE_URL = (
    "https://raw.githubusercontent.com/alibaba/clusterdata/"
    f"{SOURCE_REF}/cluster-trace-v2026-spot-gpu"
)
FILES = ("README.md", "node_info_df.csv", "job_info_df.csv")


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
        downloaded.append(
            {
                "file": name,
                "source_url": source,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "source": SOURCE,
        "source_ref": SOURCE_REF,
        "upstream_readme": f"{BASE_URL}/README.md",
        "attribution_note": (
            "Review the pinned upstream README and cited paper before reusing the dataset."
        ),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "hash_algorithm": "sha256",
        "files": downloaded,
    }
    with (args.output_dir / "source-manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Downloaded pinned trace {DATASET_VERSION}@{SOURCE_REF} to {args.output_dir}")


if __name__ == "__main__":
    main()
