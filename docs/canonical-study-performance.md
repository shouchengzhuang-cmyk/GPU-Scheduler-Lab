# Canonical study performance

This note records the measured bottleneck and the bounded run-level multiprocessing
design used by `study run`. It does not change the canonical study contract, reduce
the 180-run matrix, or turn simulated evidence into hardware evidence.

## Profiling result

Profiling used the unmodified v0.3.0 `main` implementation on Ubuntu 24.04 under
WSL2, Python 3.12.3, an Intel Core i5-13500H exposed as 16 logical CPUs, and 7.6 GiB
of memory. Four representative canonical plans covered all formal policy families,
high workload intensity, skewed GPU heterogeneity, and a history ablation.

| Plan | Wall time | Process CPU / wall |
|---|---:|---:|
| baseline / binpack | 0.51 s | 1.09 |
| workload-intensity=1.4 / fairshare-reclaim | 78.18 s | 1.09 |
| gpu-heterogeneity=skewed / topology-aware | 0.62 s | 1.09 |
| ablation-history / historical-drf | 66.30 s | 1.09 |

The combined cProfile sample recorded 578,998,655 calls. The dominant cumulative
paths were simulation CPU work, not artifact I/O:

| Hot path | Calls | Cumulative time |
|---|---:|---:|
| `Simulator.run` | 4 | 145.58 s |
| `Simulator._schedule` | 2,066 | 143.71 s |
| `Simulator._dispatch_pending` | 2,781 | 142.43 s |
| `Simulator._preempt_for` | 60,944 | 101.39 s |
| `Simulator._reclaim_for` | 60,944 | 100.76 s |
| `FairShareScheduler.prepare` | 130,587 | 50.61 s |
| `FairShareScheduler.refresh_usage` | 131,679 | 38.52 s |

The canonical contract also requests one warm-up per plan. Consequently, 180 study
plans invoke the CPU-heavy simulator 360 times. Run plans are independent and use
plan-local deterministic seeds, while aggregation is already separable from
simulation. Bounded process parallelism therefore addresses the measured bottleneck
without changing scheduler behavior.

## Concurrency design

- `--workers 1` is the default and retains serial behavior.
- `--workers N` uses at most `N` worker processes; no thread pool is used.
- Workers only execute warm-up, retry, and measured simulation calls. They do not
  write artifacts.
- The parent process is the only writer for `attempts/`, `manifest.json`, and
  `result.json`, and the only aggregation/report ordering authority.
- Stable run IDs, retry counts, resume validation, manifest identity, final sorting,
  and hash coverage are unchanged.
- CI and release workflows derive the process count from runner CPU availability and
  cap it at four.

## Full canonical benchmark

All three measurements ran sequentially on the same otherwise-idle WSL environment,
same source tree, same canonical config, and same 180-run matrix. `/usr/bin/time -v`
measured only `study run`; report rendering and verification ran afterward.

```bash
/usr/bin/time -v python -m gpu_scheduler_lab study run \
  --config study/study.yaml --workers 1
python -m gpu_scheduler_lab study report --input build/study/canonical
python -m gpu_scheduler_lab study verify --input build/study/canonical
```

The command was repeated with workers 2 and 4, moving each completed directory to an
independent benchmark location before the next run.

| Workers | Wall time | User CPU | CPU utilization | Speedup | Wall reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 50:27.46 | 3,015.87 s | 99% | 1.00x | 0.0% |
| 2 | 27:21.95 | 3,244.51 s | 197% | 1.84x | 45.8% |
| 4 | 18:43.40 | 4,372.93 s | 389% | 2.69x | 62.9% |

GNU time reported a maximum resident set metric of about 70 MiB in every group; live
observation showed each worker near 55–57 MiB plus a parent near 69 MiB. Workers=4
therefore remained well within the available memory on this host. The higher total
user CPU at four workers reflects multiprocessing overhead, heterogeneous-core
scheduling, and reduced per-core efficiency; it is why workflows cap concurrency
instead of following an unbounded CPU count.

## Equivalence and acceptance

Each group completed 180/180 runs with zero resumes, generated three tables and three
figures, and passed `study verify` over 373 hashed artifacts. Recursive directory
comparison produced no differences between workers=1, 2, and 4. The three
`hashes.sha256` files and ordered 180-run ID lists were byte-identical; no wall-clock
field exclusion was needed.

The workers=2 result reduces wall time by 45.8%, and workers=4 by 62.9%, while keeping
the complete study and evidence semantics unchanged. These timings measure this
Python discrete-event simulator only. They are not real GPU, CUDA, Kubernetes, or
production scheduler throughput evidence.
