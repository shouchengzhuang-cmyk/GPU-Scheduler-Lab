from __future__ import annotations

import sys

from gpu_scheduler_lab.cli import main

if __name__ == "__main__":
    main(["trace-import", *sys.argv[1:]])
