"""Repo-root pytest configuration.

Applies to both ``tests/`` and the root-level ``test_screener.py``.

Why this exists
---------------
Several tests exercise real pipeline functions that write to tracked,
published files as a side effect:

* ``validation/data_quality_log.csv`` -- the data-quality writer stamps rows
  into the real log.
* ``sp500_tickers.json`` -- ``factor_engine.get_sp500_tickers()`` refreshes the
  local universe cache whenever a network source succeeds.
* ``factor_output.xlsx`` -- Excel tests that load the real ``config.yaml`` pick
  up ``output.excel_file`` and rebuild the published workbook.

Those are published artifacts with provenance meaning, not test scratch. Left
alone, a plain ``pytest`` run leaves the working tree dirty, which breaks the
unattended morning routine (its clean-tree guard aborts) and risks committing
test-generated rows into the audit trail.

This fixture snapshots those files before the session and restores them after,
so running the suite is side-effect-free on the repo.

This is a guard, not isolation. The deeper fix is to point the offending tests
at ``tmp_path`` fixtures and stub the network call in
``get_sp500_tickers``. Until that lands, this keeps the tree honest.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent

# Tracked files that the suite is known to rewrite as a side effect.
PROTECTED_PATHS = [
    ROOT / "validation" / "data_quality_log.csv",
    ROOT / "sp500_tickers.json",
    ROOT / "factor_output.xlsx",
]


@pytest.fixture(scope="session", autouse=True)
def preserve_published_artifacts():
    """Restore published artifacts that the suite rewrites as a side effect."""
    saved = {}
    for path in PROTECTED_PATHS:
        saved[path] = path.read_bytes() if path.exists() else None

    yield

    for path, original in saved.items():
        if original is None:
            # Did not exist before the run; remove it if a test created it.
            if path.exists():
                path.unlink()
            continue
        if not path.exists() or path.read_bytes() != original:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
