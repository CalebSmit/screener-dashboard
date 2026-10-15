"""The suite must not write published artifacts - and must not be able to drift back into it.

Why this module exists
----------------------
``conftest.py`` snapshots the three tracked files the suite is known to rewrite and fails the
session if any of them changed (CLAUDE.md priority 8, landed 2026-10-09). That tripwire works,
but on 2026-10-14 it fired on a clean tree at 21:20 having passed at 06:40 the same day, and
the reason is the gap this module closes:

``test_screener.py``'s session-scoped ``universe_df`` fixture called
``factor_engine.get_sp500_tickers()``, which goes to the network and **rewrites the published
``sp500_tickers.json`` whenever a source succeeds**. So the write happened only when the
network was up *and* the fetched universe differed from the committed one. The 2026-10-09
per-test write detector ran on a day it did not differ, found "exactly three tests", and
missed this one entirely. A detector that runs once cannot see a conditional write; a
tripwire that fires on a condition outside the repo makes gate 1 non-deterministic, and a
non-deterministic gate 1 blocks the unattended merge for reasons no session can reproduce.

So the fixture now reads the committed file, and the loader's *offline* path is tested
directly with the network stubbed. These tests keep it that way.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The one test allowed to call the universe loader. It stubs the network and asserts the
# published file is byte-identical afterwards, which is the property that matters.
ALLOWED_CALLERS = {"test_offline_fallback_reads_the_committed_universe_and_writes_nothing"}

# Functions that rewrite a tracked, published artifact as a side effect of succeeding.
WRITES_A_PUBLISHED_ARTIFACT = {"get_sp500_tickers"}


def _test_sources() -> list[Path]:
    files = sorted((ROOT / "tests").glob("test_*.py"))
    files.append(ROOT / "test_screener.py")
    return [f for f in files if f.exists()]


def _enclosing_function(tree: ast.AST, node: ast.AST) -> str | None:
    """Name of the innermost function definition containing ``node``."""
    best = None
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                if child is node:
                    # Deeper definitions win, so record the smallest containing span.
                    if best is None or parent.lineno >= best[1]:
                        best = (parent.name, parent.lineno)
                    break
    return best[0] if best else None


def _calls_to(name: str) -> list[tuple[Path, int, str | None]]:
    found = []
    for path in _test_sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - a syntax error fails elsewhere, loudly
            pytest.fail(f"{path.name} does not parse: {exc}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            target = (func.attr if isinstance(func, ast.Attribute)
                      else func.id if isinstance(func, ast.Name) else None)
            if target == name:
                found.append((path, node.lineno, _enclosing_function(tree, node)))
    return found


@pytest.mark.parametrize("name", sorted(WRITES_A_PUBLISHED_ARTIFACT))
def test_only_the_stubbed_test_calls_a_loader_that_writes_a_published_file(name):
    offenders = [(p.name, line, fn) for p, line, fn in _calls_to(name)
                 if fn not in ALLOWED_CALLERS]
    assert not offenders, (
        f"{name}() rewrites a published artifact when the network succeeds, so the suite may "
        f"only call it from {sorted(ALLOWED_CALLERS)} (which stubs the network and asserts the "
        f"file is unchanged). Read the committed file instead. Offenders: {offenders}"
    )


def test_the_allowed_caller_still_exists():
    """Guards against the allowlist outliving the test it names."""
    callers = {fn for _p, _l, fn in _calls_to("get_sp500_tickers")}
    assert callers & ALLOWED_CALLERS, (
        "ALLOWED_CALLERS names a test that no longer calls get_sp500_tickers - either the "
        "offline-fallback test was renamed or deleted, and the loader is now untested."
    )


def test_the_universe_fixture_does_not_go_to_the_network():
    """The fixture is session-scoped, so its write would be attributed to an unrelated test."""
    tree = ast.parse((ROOT / "test_screener.py").read_text(encoding="utf-8"))
    fixture = next((n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "universe_df"), None)
    assert fixture is not None, "test_screener.py no longer defines the universe_df fixture"
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else
              n.func.id if isinstance(n.func, ast.Name) else None
              for n in ast.walk(fixture) if isinstance(n, ast.Call)}
    assert not (called & WRITES_A_PUBLISHED_ARTIFACT), (
        f"universe_df calls {called & WRITES_A_PUBLISHED_ARTIFACT}; it must read the committed "
        "sp500_tickers.json instead. See this module's docstring."
    )


def test_the_conftest_tripwire_still_guards_all_three_files():
    """A session that 'fixes' a red gate by shortening this list would hide the next one."""
    import conftest

    names = {p.name for p in conftest.PROTECTED_PATHS}
    assert {"sp500_tickers.json", "data_quality_log.csv", "factor_output.xlsx"} <= names, (
        f"conftest.PROTECTED_PATHS no longer covers all three published artifacts: {names}"
    )


def test_the_conftest_tripwire_fails_and_does_not_merely_restore():
    """The guard must *fail* the session, not quietly clean up (conftest.py, 2026-10-09)."""
    src = (ROOT / "conftest.py").read_text(encoding="utf-8")
    assert "pytest.fail(" in src, (
        "conftest's artifact guard no longer fails the session. Restoring the file without "
        "failing is how a test that writes a published artifact goes unnoticed for months."
    )
