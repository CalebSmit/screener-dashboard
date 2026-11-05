"""The research engine's published rows were produced by the code they are published beside.

Rule 7 of `plan/research-engine.md`: *"A result names the code that produced it, or it is not a
result."* Every ledger row does name it - `code_hash` over all of `v2_experiments.RESULT_PATHS`,
`content_hash` over the monthly records, `base_config_hash` over the baseline weighting,
`method_version` over how the comparison is computed. **Nothing compared those names with the
live tree.** `test_every_ledger_row_the_report_reads_names_its_code_and_its_baseline` asserts the
newest rows agree *with each other*, which is a different fact, and
`test_the_baseline_weighting_is_hashed_separately_from_the_transformed_one` reads
`base_config_hash`, which moves only when `config.yaml`'s **parsed** contents move.

Measured 2026-11-05, which is why this module exists: **all 32 rows had been stale since
2026-11-03**, when `universe_history.py` - a `RESULT_PATHS` file - was edited by the universe
phase-1 slice, and `tests/test_v2_experiments.py` reported **57 passed** throughout. Two data runs
published `research.html` from those rows in between. (The rerun moved nothing - 32 ids, 27,225
statistics, 0 differences, `research/measurements/2026-11-05-ledger-rerun-moved-nothing.py` - but
that is a fact about which file moved, established afterwards, and is exactly what the hash exists
so that nobody has to assume.)

**Why its own module, and not `tests/test_v2_experiments.py`.** That module is named by claim
`overview.research_engine`, so the 02:00 publish gate runs it and a failure **discards the night's
data** (`CLAUDE.md`, the "data->red" row). A stale ledger is caused by a *code* session editing a
result-affecting file and not rerunning; it should block that session's merge, not throw away a
night of evidence collected by the data loop. No claim names this module, and the last test here
fails if one ever does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import v2_experiments as X        # noqa: E402

REL = "tests/test_ledger_is_level_with_the_code.py"
FIX = ("python scripts/rerun_experiments.py --jobs 4   # then `python v2_experiments.py report`, "
       "and commit ledger.jsonl, RESULTS.md, summary.json and research.html")


def _records_present() -> bool:
    """The engine's own inputs, without which the live `content_hash` means nothing."""
    import backtest_v2 as v2
    if not any(X.METRICS_DIR.glob("*.parquet") if hasattr(X, "METRICS_DIR")
               else v2.CACHE.glob("v2_metrics/*.parquet")):
        return False
    return all((ROOT / p).exists() for p in X.DATA_SOURCES.values())


@pytest.fixture(scope="module")
def newest() -> dict:
    if not X.LEDGER.exists():
        pytest.skip(f"MISSING DATA: {X.LEDGER.relative_to(ROOT)} - nothing has been run here")
    if not _records_present():
        pytest.skip("MISSING DATA: the v2 monthly records or their sources are absent "
                    "(gitignored); the live content_hash is not comparable")
    import rerun_experiments as R
    rows = R.latest_rows(X.LEDGER)
    if not rows:
        pytest.skip("MISSING DATA: the ledger is empty")
    return rows


@pytest.fixture(scope="module")
def live() -> dict:
    import rerun_experiments as R
    return R.current_provenance(X)


@pytest.mark.parametrize("key", ["code_hash", "content_hash", "base_config_hash", "method_version"])
def test_every_published_row_was_produced_by_the_committed_code(newest, live, key):
    """One row per key, so a failure says *which* kind of staleness it is."""
    stale = {i: r.get(key) for i, r in newest.items() if r.get(key) != live[key]}
    assert not stale, (
        f"{len(stale)} of {len(newest)} newest ledger rows carry {key}={sorted(set(stale.values()))} "
        f"against the live {key}={live[key]!r}. research.html and the methodology page quote these "
        f"rows, so they state results produced by code that has since changed (rule 3, rule 7).\n"
        f"Fix: {FIX}\nStale ids: {sorted(stale)}")


def test_the_four_keys_are_the_ones_the_rerun_tool_levels(live):
    """One definition of "current", shared with `scripts/rerun_experiments.py` - so this cannot
    pass while `--status` says stale, which is how the gap went unseen."""
    import rerun_experiments as R
    assert set(live) == set(R.PROVENANCE_KEYS)
    assert R.PROVENANCE_KEYS == ("code_hash", "content_hash", "base_config_hash", "method_version")


def test_the_live_fingerprint_moves_when_a_result_affecting_file_moves(tmp_path):
    """The teeth: the hash this module compares against really does see every `RESULT_PATHS` file.

    Checked by content rather than by editing the tree - `code_fingerprint(root)` hashes the files
    under `root`, so a copy with one file changed must produce a different `code_hash`, for **each**
    of the eight files in turn. `universe_history.py` is one of them, and was the file nothing saw.
    """
    for rel in X.RESULT_PATHS:
        src = ROOT / rel
        if not src.is_file():                                  # pragma: no cover - full tree only
            pytest.skip(f"MISSING DATA: {rel}")
        root = tmp_path / rel.replace("/", "_").replace(".", "_")
        for other in X.RESULT_PATHS:
            dst = root / other
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT / other).read_bytes())
        before = X.code_fingerprint(root)["code_hash"]
        (root / rel).write_bytes((ROOT / rel).read_bytes() + b"\n# moved\n")
        assert X.code_fingerprint(root)["code_hash"] != before, rel
        assert before == X.code_fingerprint()["code_hash"], (
            f"the copy of the tree is not the real one for {rel}")


def test_this_module_stays_out_of_the_data_loops_publish_gate():
    """A stale ledger must not discard a night's data - see this module's docstring.

    The gate's membership is derived (`scripts/check_published_claims.module_paths`): the curated
    list plus every module a registered claim's `checked_by` names. Registering a claim against a
    test here would quietly couple the data loop to a code session's missed rerun.
    """
    import check_published_claims as gate
    assert REL not in gate.module_paths(), (
        f"{REL} is now in the 02:00 publish gate. A stale ledger would discard the night's data. "
        "Put the check in a module the gate runs only if that is what you mean to do, and say so.")
