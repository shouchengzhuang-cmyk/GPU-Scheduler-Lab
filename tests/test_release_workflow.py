from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_release_prepare_budget_covers_the_full_canonical_study() -> None:
    workflow = (ROOT / ".github/workflows/publish-release.yml").read_text(encoding="utf-8")
    prepare, remainder = workflow.split("\n  publish:", maxsplit=1)
    _publish, notification = remainder.split("\n  notify-prepare-failure:", maxsplit=1)

    assert "timeout-minutes: 120" in prepare
    assert "min(4, os.cpu_count() or 1)" in prepare
    assert 'make reproduce-study STUDY_WORKERS="${study_workers}"' in prepare
    assert "needs.prepare.result == 'failure'" in notification
    assert "needs.prepare.result == 'cancelled'" in notification
    assert "PREPARE_RESULT: ${{ needs.prepare.result }}" in notification
    assert r"result \`${PREPARE_RESULT}\`" in notification
