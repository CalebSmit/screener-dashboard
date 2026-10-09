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

**Isolation landed 2026-10-09** (CLAUDE.md priority 8). Measured that day with a
per-test write detector over the whole suite: exactly three tests wrote these files -
``TestDQLog::test_flush_writes_csv`` (the log), ``TestFullPipeline::test_end_to_end``
(the workbook, and a synthetic scored table into the real ``cache/``) and
``TestFullPipeline::test_tiny_pipeline`` (``get_sp500_tickers`` over the network). All
three now write to ``tmp_path`` or read the committed universe.

So the guard is now a **tripwire**: it still restores the files, and then fails the
session, naming the file, so a new test that writes a published artifact is caught the
day it is written instead of being silently cleaned up for months.
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

    written = []
    for path, original in saved.items():
        if original is None:
            # Did not exist before the run; remove it if a test created it.
            if path.exists():
                path.unlink()
                written.append(path.name)
            continue
        if not path.exists() or path.read_bytes() != original:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
            written.append(path.name)
    if written:
        pytest.fail("A test wrote published artifacts (restored, but the test must use tmp_path "
                    f"or stub the network): {', '.join(written)}. See conftest.py.", pytrace=False)
