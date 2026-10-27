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

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent

# Tracked files that the suite is known to rewrite as a side effect.
PROTECTED_PATHS = [
    ROOT / "validation" / "data_quality_log.csv",
    ROOT / "sp500_tickers.json",
    ROOT / "factor_output.xlsx",
    # Written only by the 02:00 data loop, after every gate (scripts/write_signals.py,
    # 2026-10-24). A test that writes it would be asserting a run was gated when it
    # was not, and the bot trades this file - so it is a tripwire here from the day
    # it was introduced rather than after the first accident.
    ROOT / "signals.json",
]


_FAIL_MESSAGE = ("A test wrote published artifacts (restored, but the test must use tmp_path "
                 "or stub the network): {}. See conftest.py.")


# ---------------------------------------------------------------------------
# Under pytest-xdist (2026-10-26). The suite runs in parallel where xdist is
# installed (scripts/run_tests.py), and a per-worker snapshot would be wrong
# there: a worker that starts after another worker's test wrote a file would
# snapshot the *written* bytes and later "restore" them. So under xdist the
# controller snapshots once, before any worker exists, and hands every worker
# the digests; each worker fails its session if a file no longer matches them
# (the writer's worker always does, because its teardown follows its own
# write), and the controller restores the originals once every worker is done
# and fails the run itself as a second line. Serially (no xdist, or -n 0) the
# fixture below behaves exactly as it always has. tests/test_parallel_suite.py.
# ---------------------------------------------------------------------------

def _digest(data):
    return None if data is None else hashlib.sha256(data).hexdigest()


def _is_xdist_controller(config):
    return config.pluginmanager.getplugin("dsession") is not None


@pytest.hookimpl(trylast=True)  # after xdist's own pytest_configure registers "dsession"
def pytest_configure(config):
    if not hasattr(config, "workerinput") and _is_xdist_controller(config):
        config._published_snapshot = {
            str(p): (p.read_bytes() if p.exists() else None) for p in PROTECTED_PATHS
        }


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """xdist hook, run in the controller for each worker before it starts."""
    snap = getattr(node.config, "_published_snapshot", None)
    if snap is not None:
        node.workerinput["published_artifact_digests"] = {k: _digest(v) for k, v in snap.items()}


def pytest_sessionfinish(session, exitstatus):
    snap = getattr(session.config, "_published_snapshot", None)
    if snap is None:
        return
    written = []
    for name, original in snap.items():
        path = Path(name)
        if original is None:
            if path.exists():
                path.unlink()
                written.append(path.name)
        elif not path.exists() or path.read_bytes() != original:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
            written.append(path.name)
    if written:
        session.config._published_written = written
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    written = getattr(config, "_published_written", None)
    if written:
        terminalreporter.write_sep("=", "published artifacts written", red=True)
        terminalreporter.write_line(_FAIL_MESSAGE.format(", ".join(written)))


@pytest.fixture(scope="session", autouse=True)
def preserve_published_artifacts(request):
    """Restore published artifacts that the suite rewrites as a side effect."""
    workerinput = getattr(request.config, "workerinput", None)
    if workerinput is not None and "published_artifact_digests" in workerinput:
        # An xdist worker: check against the controller's snapshot; the
        # controller restores (see above), so workers never write these files.
        expected = workerinput["published_artifact_digests"]
        yield
        changed = [Path(name).name for name, digest in expected.items()
                   if _digest(Path(name).read_bytes() if Path(name).exists() else None) != digest]
        if changed:
            pytest.fail(_FAIL_MESSAGE.format(", ".join(changed)), pytrace=False)
        return

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
        pytest.fail(_FAIL_MESSAGE.format(", ".join(written)), pytrace=False)
