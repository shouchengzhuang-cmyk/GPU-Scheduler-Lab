# Code review remediation plan

This document turns the 2026-09-05 code review into a bounded correctness plan. The goal is to improve evidence quality without reopening feature scope.

## Goal

Make canonical study metrics describe one stable quantity across all job classes, and preserve a clear boundary between simulator evidence and future follow-up work.

## Tasks

### P0 — Canonical waiting-time semantics

- Define `waiting_time` uniformly as `first_start_time - arrival_time` for every completed job.
- Keep preemption, checkpoint, restart, elastic execution, and later suspension effects in turnaround and explicit overhead metrics rather than folding them into queue wait.
- Add a regression test proving fixed and elastic jobs use the same definition.
- Update the canonical metric definition so `p95_waiting_time` is explicitly queue delay.

Acceptance criteria:

- Fixed and elastic jobs with the same arrival and first-start timestamps report the same waiting time regardless of completion time.
- `pytest`, Ruff, mypy, study contract validation, invariant validation, and the small canonical study CI remain green.
- Existing canonical results that contain `p95_waiting_time` are treated as stale after this semantic change.

### P0 — Refresh canonical evidence after merge

- Re-run the canonical study from the merged commit.
- Regenerate summaries, report, charts, manifests, hashes, and any golden evidence that depends on waiting-time values.
- Do not compare the refreshed p95 values numerically with the previous baseline as though the metric definition were unchanged.

Acceptance criteria:

- `make reproduce-study` succeeds on the merged commit.
- `study verify` accepts the generated bundle.
- Reports state the new queue-wait definition.

### P1 — Make `ANY` vendor selection explicit

Current generic multi-accelerator selection is deterministic but can inherit vendor lexical ordering. A follow-up should compare feasible vendor pools by scheduler score and use a documented deterministic tie-break rather than accidental enum/string order.

Acceptance criteria:

- `ANY` has a documented tie-break contract.
- Tests prove changing enum declaration or lexical order does not silently change placement preference.

### P2 — Harden elastic resize invariants

Move replica-bound validation into the domain boundary (`Cluster.resize`) in addition to policy-level checks so invalid replica counts cannot be committed by a future scheduler implementation.

Acceptance criteria:

- Resize below `min_replicas` or above `max_replicas` fails at the cluster mutation boundary.
- Existing elastic scheduling tests remain green.

## Scope boundary

This remediation is not a new simulator phase. P0 is release-blocking for trustworthy canonical latency evidence. P1 and P2 are follow-ups and should be separate PRs unless a later review finds a direct correctness dependency.
