"""The page says which run, which code and which raw fetch it came from.

`OWNER_FOCUS.md` 2026-10-21 item 6 / screener-bot ROADMAP 1.10. See
``run_fingerprint.py`` for why each of the four identifiers is there. The
property these tests exist to defend is narrow and easy to lose:

* the raw hash is of **this run's own** fetch, never the fallback copy the
  generator borrows from a neighbouring run for display, and
* a missing identifier is published as ``null``, never as a plausible-looking
  substitute ("unknown", another run's id, today's date).

A provenance field that quietly reports the wrong thing is worse than one that
is absent, because the absent one cannot be mistaken for evidence.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_fingerprint as F  # noqa: E402


@pytest.fixture()
def run(tmp_path):
    d = tmp_path / "runs" / "abc123def456"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "run_id": "abc123def456",
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
        "methodology_version": "cafe1234",
        "start_time": "2026-10-24T02:00:00",
    }), encoding="utf-8")
    (d / F.RAW_FETCH_NAME).write_bytes(b"raw fetch bytes")
    return d


def test_the_four_identifiers_are_read_from_the_run(run):
    fp = F.fingerprint(run)
    assert fp["run_id"] == "abc123def456"
    assert fp["git_sha"] == "0123456789abcdef0123456789abcdef01234567"
    assert fp["methodology_version"] == "cafe1234"
    assert fp["raw_sha256"] == hashlib.sha256(b"raw fetch bytes").hexdigest()


def test_the_hash_is_the_one_sha256sum_prints(run):
    """The archived copy is verified with an ordinary sha256sum, so chunking
    must not change the digest."""
    expected = hashlib.sha256((run / F.RAW_FETCH_NAME).read_bytes()).hexdigest()
    assert F.sha256_file(run / F.RAW_FETCH_NAME) == expected


def test_a_large_file_hashes_the_same_across_the_chunk_boundary(tmp_path):
    big = tmp_path / "big.bin"
    blob = bytes(range(256)) * 9000  # > 2 MB, so more than two 1 MB chunks
    big.write_bytes(blob)
    assert F.sha256_file(big) == hashlib.sha256(blob).hexdigest()


def test_a_cache_hit_run_reports_no_raw_hash_rather_than_a_neighbours(run):
    """The regression this is here for.

    ``generate_dashboard._find_raw_fetch`` falls back to the newest other run
    that has a raw fetch, so a cache-hit run can still show price targets. That
    is right for display. Borrowing it here would stamp the page with the
    fingerprint of a fetch this run never made.
    """
    sibling = run.parent / "other999"
    sibling.mkdir()
    (sibling / F.RAW_FETCH_NAME).write_bytes(b"a different run's fetch")
    (run / F.RAW_FETCH_NAME).unlink()

    assert F.raw_fetch_sha256(run) is None
    assert F.fingerprint(run)["raw_sha256"] is None


def test_git_unknown_is_published_as_absent_not_as_a_commit(run):
    """``run_context._get_git_sha`` returns the literal string "unknown" when it
    cannot reach git. Published as-is it reads like a value."""
    meta = json.loads((run / "meta.json").read_text())
    meta["git_sha"] = "unknown"
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert F.fingerprint(run)["git_sha"] is None


def test_a_run_with_no_metadata_still_fingerprints(tmp_path):
    d = tmp_path / "halfbuilt"
    d.mkdir()
    fp = F.fingerprint(d)
    assert fp["run_id"] == "halfbuilt"       # the folder name is the run id
    assert fp["git_sha"] is None
    assert fp["methodology_version"] is None
    assert fp["raw_sha256"] is None


def test_unreadable_metadata_is_absent_not_fatal(tmp_path):
    d = tmp_path / "corrupt"
    d.mkdir()
    (d / "meta.json").write_text("{not json", encoding="utf-8")
    assert F.fingerprint(d)["git_sha"] is None


def test_the_generator_publishes_the_fingerprint_it_reads(tmp_path, monkeypatch):
    """End to end through ``load_run_data`` -> ``prepare_dashboard_data``:
    whatever the fingerprint says is what lands in ``kpis``."""
    import pandas as pd
    import generate_dashboard as g

    d = tmp_path / "run7"
    d.mkdir()
    df = pd.DataFrame({
        "Ticker": ["AAA"], "Company": ["Alpha"], "Sector": ["Energy"],
        "Composite": [50.0], "Rank": [1],
        "Value_Trap_Flag": [False], "Growth_Trap_Flag": [False],
    })
    df.to_parquet(d / "05_final_scored.parquet")
    # A real parquet: load_run_data reads it for the display fields.
    pd.DataFrame({"Ticker": ["AAA"], "currentPrice": [10.0]}).to_parquet(d / F.RAW_FETCH_NAME)
    raw_bytes = (d / F.RAW_FETCH_NAME).read_bytes()
    (d / "meta.json").write_text(json.dumps({
        "run_id": "run7", "git_sha": "deadbeef" * 5,
        "methodology_version": "v9", "start_time": "2026-10-24T02:00:00",
    }), encoding="utf-8")

    payload = json.loads(g.prepare_dashboard_data(g.load_run_data(d)))
    kpis = payload["kpis"]
    assert kpis["run_id"] == "run7"
    assert kpis["git_sha"] == "deadbeef" * 5
    assert kpis["methodology_version"] == "v9"
    assert kpis["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
