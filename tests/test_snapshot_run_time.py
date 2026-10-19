"""Every new improvement snapshot records when its run started (2026-10-19).

A snapshot used to record its date and not its time, so the track record had to infer from the
run's recorded prices whether it had already seen that day's close
(``track_record.runs_after_the_close``; ``research/2026-10-19-track-record.md`` section 6 - the
2026-08-10 rebalance ran at 19:22 CDT and had entered at a close set before its ranking existed).
``record_run_snapshot`` now stores the run context's start time as ``run_started_at``, ISO 8601
with its UTC offset, and the track record reads it in preference to the inference. Old snapshots
keep the inference; ``tests/test_track_record_claims.py`` holds both paths.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import improvement_engine as ie  # noqa: E402
import track_record as tr  # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    monkeypatch.setattr(ie, "SNAPSHOTS_DIR", snaps)
    monkeypatch.setattr(ie, "DISPERSION_HISTORY_PATH", tmp_path / "dispersion_history.csv")
    # the evidence files downstream of a snapshot are not this module's subject: keep them untouched
    monkeypatch.setattr(ie, "compute_forward_returns", lambda *a, **k: None)
    monkeypatch.setattr(ie, "compute_live_ic", lambda *a, **k: None)
    monkeypatch.setattr(ie, "LIVE_IC_HISTORY_PATH", tmp_path / "live_ic_history.csv")
    return snaps


def _scored(n=6):
    return pd.DataFrame({
        "Ticker": [f"T{i}" for i in range(n)],
        "Sector": ["Information Technology"] * n,
        "Composite": [50.0 + i for i in range(n)],
        "Rank": list(range(1, n + 1)),
        "_current_price": [100.0 + i for i in range(n)],
        **{c: [50.0 + i for i in range(n)] for c in ie.CATEGORY_SCORES},
    })


def test_the_snapshot_records_the_run_start_with_its_offset(isolated):
    started = datetime(2026, 10, 19, 2, 0, 4, tzinfo=timezone(timedelta(hours=-5)))
    path = ie.record_run_snapshot("runaaaa", "2026-10-19", _scored(), None, {}, run_started=started)
    snap = pd.read_parquet(path)
    assert snap["run_started_at"].nunique() == 1
    assert snap["run_started_at"].iloc[0] == "2026-10-19T02:00:04-05:00"
    assert tr.run_started_at(path) == "2026-10-19T02:00:04-05:00"


def test_a_naive_start_is_stamped_with_this_machines_offset(isolated):
    """``RunContext.start_time`` is ``datetime.now()`` - naive local time."""
    naive = datetime(2026, 10, 19, 2, 0, 4)
    path = ie.record_run_snapshot("runaaaa", "2026-10-19", _scored(), None, {}, run_started=naive)
    stamp = pd.read_parquet(path)["run_started_at"].iloc[0]
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None
    assert parsed == naive.astimezone()            # the same instant, now unambiguous


def test_an_iso_string_from_meta_json_is_accepted():
    assert ie.run_started_iso("2026-10-19T02:00:04.793163-05:00") == "2026-10-19T02:00:04.793163-05:00"
    assert datetime.fromisoformat(ie.run_started_iso("2026-10-19T02:00:04.793163")).tzinfo is not None
    assert ie.run_started_iso(None) is None
    assert ie.run_started_iso("") is None
    assert ie.run_started_iso("garbage") is None


def test_without_a_start_time_the_snapshot_is_as_before(isolated):
    path = ie.record_run_snapshot("runaaaa", "2026-10-19", _scored(), None, {})
    assert "run_started_at" not in pd.read_parquet(path).columns
    assert tr.run_started_at(path) is None


def test_a_recorded_evening_run_is_read_as_after_the_close(isolated):
    """End to end: the snapshot the engine writes is what the track record reads."""
    started = datetime(2026, 8, 10, 19, 22, tzinfo=timezone(timedelta(hours=-5)))   # CDT
    ie.record_run_snapshot("runaaaa", "2026-08-10", _scored(), None, {}, run_started=started)
    prices = pd.DataFrame({f"T{i}": [100.0 + i, 101.0 + i, 102.0 + i] for i in range(6)},
                          index=pd.to_datetime(["2026-08-07", "2026-08-10", "2026-08-11"]))
    assert tr.runs_after_the_close(["2026-08-10"], prices, snap_dir=isolated) == ["2026-08-10"]
    assert tr.untimed_runs(["2026-08-10"], snap_dir=isolated) == []


def test_the_pipeline_passes_the_run_context_start_time():
    src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    step13 = src[src.index("# ---- 13. Improvement tracking snapshot"):]
    call = step13[step13.index("record_run_snapshot(ctx.run_id"):]
    assert "run_started=ctx.start_time" in call[:200]


def test_the_backfill_records_each_runs_own_start(tmp_path, monkeypatch):
    """A backfilled snapshot is written weeks later; the time it carries is the run's, from
    its meta.json, not the backfill's."""
    snaps = tmp_path / "improvement" / "snapshots"
    snaps.mkdir(parents=True)
    run = tmp_path / "runs" / "abc123"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"run_id": "abc123",
                                               "start_time": "2026-02-20T19:05:00-06:00"}))
    _scored().to_parquet(run / "05_final_scored.parquet", index=False)
    monkeypatch.setattr(ie, "ROOT", tmp_path)
    monkeypatch.setattr(ie, "SNAPSHOTS_DIR", snaps)
    monkeypatch.setattr(ie, "compute_forward_returns", lambda *a, **k: None)
    assert ie.backfill_from_existing_runs() == 1
    snap = pd.read_parquet(snaps / "2026-02-20_abc123.parquet")
    assert snap["run_started_at"].iloc[0] == "2026-02-20T19:05:00-06:00"
