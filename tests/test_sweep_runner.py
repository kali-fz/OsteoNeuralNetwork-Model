"""The sweep driver's bookkeeping, which is where its silent failures live.

`overnight_sweep.py` runs unattended for hours and its output is a comparison
table someone then promotes a model from. That makes a *wrong row* far more
dangerous than a crash: a crash costs a night, a wrong row costs a promotion.

Both tests below pin a mistake that actually happened on 2026-09-04 and that
produced no error of any kind.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_sweep():
    """Import the driver by path -- it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location(
        "overnight_sweep", REPO_ROOT / "scripts" / "overnight_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep():
    return _load_sweep()


def _make_run(reports: Path, name: str) -> Path:
    directory = reports / name
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text("{}", encoding="utf-8")
    return directory


def test_a_run_name_does_not_swallow_one_that_extends_it(sweep, tmp_path):
    """`r384-w050` must not match `r384-w050-aug`.

    THE BUG THIS PINS
    -----------------
    `_newest_run` globbed `{tag}-*` and took the most recently written match.
    `r384-w050-aug` matches `r384-w050-*`, and it was written later, so the
    `r384-w050` row of the comparison table silently reported the aug run's
    numbers. The table printed the aug results twice under two different names
    and the real r384-w050 row disappeared. Nothing raised, and the two rows
    were individually plausible.

    Naming one run as another plus a suffix is the natural way to express "same
    recipe, plus augmentation", so the driver has to cope rather than the plan
    having to avoid it.
    """
    reports = tmp_path / "reports"
    plain = _make_run(reports, "sweep-S-r384-w050-20260904-193722")
    augmented = _make_run(reports, "sweep-S-r384-w050-aug-20260904-201307")
    # The aug run is the newer of the two, which is what made the bug bite.
    import os
    import time

    os.utime(augmented, (time.time() + 60, time.time() + 60))

    assert sweep._newest_run("sweep-S-r384-w050", reports) == plain
    assert sweep._newest_run("sweep-S-r384-w050-aug", reports) == augmented


def test_a_directory_without_a_timestamp_is_not_a_run(sweep, tmp_path):
    """Only `{tag}-YYYYmmdd-HHMMSS` counts.

    Report directories accumulate siblings -- archived copies, notes, whatever a
    person leaves beside them. Matching on the stamp shape rather than on any
    suffix keeps those out of the table.
    """
    reports = tmp_path / "reports"
    real = _make_run(reports, "sweep-S-w025-20260904-150749")
    _make_run(reports, "sweep-S-w025-backup")
    _make_run(reports, "sweep-S-w025-20260904-150749-copy")

    assert sweep._runs_for("sweep-S-w025", reports) == [real]


def test_an_absent_tag_resolves_to_nothing_rather_than_guessing(sweep, tmp_path):
    """A missing run is reported as missing, not silently matched to a sibling."""
    reports = tmp_path / "reports"
    _make_run(reports, "sweep-S-w010-20260904-084858")

    assert sweep._newest_run("sweep-S-w999", reports) is None
    assert sweep._runs_for("sweep-S-w999", reports) == []


def test_the_completion_artefact_for_training_is_summary_not_best(sweep):
    """`best.pt` is rewritten every improving epoch; `summary.json` is written once.

    Polling `best.pt` would declare training finished at the first epoch that
    improved on the initial score, which on these runs is epoch one. The
    distinction is the whole reason training can be detected by artefact at all,
    so it is pinned rather than left to a comment.
    """
    source = (REPO_ROOT / "scripts" / "overnight_sweep.py").read_text(encoding="utf-8")
    marker = 'return max(fresh, key=lambda p: p.stat().st_mtime) / "summary.json"'
    assert marker in source, (
        "training completion is no longer detected via summary.json. If this "
        "moved to best.pt, training will be cut short at the first improving "
        "epoch -- see _run's docstring."
    )
