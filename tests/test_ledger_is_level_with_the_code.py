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

**What `content_hash` is taken over, corrected 2026-11-10.** It hashed all five
`DATA_SOURCES`, and the 02:00 data loop refreshes `data/sec/pit/facts.parquet` weekly
(`sec_fundamentals.FACTS_MAX_AGE_DAYS = 7`). It did so at 02:16 that morning, all 32 rows went
stale, and this module turned `main` red with no commit touching any file it names - which is the
"data->red" row arriving in the *code* loop, where the cost is a blocked merge rather than a
discarded night. Measured at that commit: `month_files` identical (`21d6dd1cf5f34531`),
`month_file_count` 176, `base_run` identical, and `v2_daily`, `v2_index`, `splits` and
`sec_facts_predecessor` identical - `sec_facts` alone had moved. The filings reach a row only
through the monthly records, which `month_files` already hashes, so no number could have changed
and the prescribed rerun would have re-stamped the rows while reading byte-identical inputs. The
verdict now reads `v2_experiments.DECIDING_PARTS`; the other sources stay in every row's
`content_parts` and are reported rather than alarmed on, which is the rule `provenance_note`
already won for `code_version` against `code_hash` on 2026-10-18.

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


def _content_root(tmp_path: Path, name: str) -> Path:
    """A copy of the tree the content fingerprint reads: the five sources and one month file."""
    root = tmp_path / name
    for rel in X.DATA_SOURCES.values():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = ROOT / rel
        dst.write_bytes(src.read_bytes() if src.is_file() else b"x")
    months = root / "cache" / "v2_metrics"
    months.mkdir(parents=True, exist_ok=True)
    (months / "2020-01.parquet").write_bytes(b"month")
    (months / "_build.json").write_text('{"base_run": "r1"}', encoding="utf-8")
    return root


def test_a_source_read_at_run_time_moves_the_verdict(tmp_path):
    """The teeth of the split, first half: a file an experiment actually reads must still stale
    every row. `v2_daily` is the prices every forward return is computed from."""
    for name in X.SOURCES_READ_AT_RUN:
        root = _content_root(tmp_path, f"run_{name}")
        months = root / "cache" / "v2_metrics"
        before = X.content_fingerprint(root, months)["content_hash"]
        p = root / X.DATA_SOURCES[name]
        p.write_bytes(p.read_bytes() + b"moved")
        assert X.content_fingerprint(root, months)["content_hash"] != before, name


def test_a_build_only_source_is_recorded_and_does_not_stale_a_row(tmp_path):
    """The teeth of the split, second half - and the only thing stopping the weekly red from
    coming back. A build input that moved while the monthly records did not cannot have changed
    a number, so the verdict must not move; but the row must still name the file it read, or the
    provenance this engine exists to keep has been dropped rather than rescoped (rule 7)."""
    build_only = [k for k in X.DATA_SOURCES if k not in X.SOURCES_READ_AT_RUN]
    assert "sec_facts" in build_only, "the file that caused this is back in the verdict"
    for name in build_only:
        root = _content_root(tmp_path, f"build_{name}")
        months = root / "cache" / "v2_metrics"
        before = X.content_fingerprint(root, months)
        p = root / X.DATA_SOURCES[name]
        p.write_bytes(p.read_bytes() + b"moved")
        after = X.content_fingerprint(root, months)
        assert after["content_hash"] == before["content_hash"], name
        assert after["content_parts"][name] != before["content_parts"][name], (
            f"{name} is no longer recorded in content_parts - the row cannot name what it read")


def test_a_rebuilt_month_still_stales_every_row(tmp_path):
    """Why the split is safe: a changed build input matters only once the months are rebuilt, and
    that is what `month_files` sees. Without this, the split would be a hole."""
    root = _content_root(tmp_path, "months")
    months = root / "cache" / "v2_metrics"
    before = X.content_fingerprint(root, months)["content_hash"]
    (months / "2020-01.parquet").write_bytes(b"rebuilt")
    assert X.content_fingerprint(root, months)["content_hash"] != before
    (months / "2020-02.parquet").write_bytes(b"another")       # and a month gained or lost
    assert X.content_fingerprint(root, months)["content_hash"] != before


def test_every_deciding_part_is_one_the_fingerprint_computes():
    """`DECIDING_PARTS` names keys of `content_parts`; a renamed source must not drop out of the
    verdict silently, which would be this defect inverted and far worse."""
    parts = X.content_fingerprint()["content_parts"]
    assert set(X.DECIDING_PARTS) <= set(parts), sorted(set(X.DECIDING_PARTS) - set(parts))
    assert set(X.SOURCES_READ_AT_RUN) <= set(X.DATA_SOURCES), "a source read at run time is unhashed"


def test_a_build_input_that_moved_is_reported_not_hidden():
    """Information, not an alarm - `provenance_note`'s own rule, on two rows that differ only in
    the filings cache they were built from. The note must name the file and must not tell the
    reader to rerun, because rerunning would reproduce the same numbers."""
    base = {"code_version": "v1", "code_hash": "c1", "base_config_hash": "b1",
            "content_hash": "h1", "content_parts": {"month_files": "m", "sec_facts": "old"}}
    moved = {**base, "content_parts": {"month_files": "m", "sec_facts": "new"}}
    assert X._build_inputs_moved([base, moved]) == ["sec_facts"]
    assert X._build_inputs_moved([base, base]) == []
    note = X.provenance_note([base, moved])
    assert "sec_facts" in note and "no number moved" in note, note
    assert "rerun" not in note.lower(), note
    assert not note.startswith("**"), note          # not an alarm


def test_the_filings_cache_moving_overnight_does_not_stale_the_ledger(newest):
    """The case as it happened, reported from the real ledger: which recorded build inputs have
    moved since these rows ran, and that the verdict is unaffected by them."""
    rows = list(newest.values())
    row_parts = rows[0].get("content_parts") or {}
    if not row_parts:
        pytest.skip("MISSING DATA: the newest rows are pre-8b and record no content_parts")
    live_parts = X.content_fingerprint()["content_parts"]
    moved = sorted(k for k, v in live_parts.items()
                   if k in row_parts and row_parts[k] != v)
    print(f"parts that moved since these rows ran: {moved or 'none'}")
    assert not [k for k in moved if k in X.DECIDING_PARTS], (
        f"a part the verdict reads has moved: {moved}. Fix: {FIX}")


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
