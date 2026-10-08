"""Two runs on one day are one observation, not two.

WHY THIS EXISTS - CLAUDE.md priority 0.6, open since 2026-08-11 and closed 2026-10-08.

A warm-started run writes an improvement-engine snapshot like any other, so a day with a
scheduled 02:00 run and a second run later produced more "observations" than there were
real data points behind them. `compute_forward_returns` already collapsed several snapshots
of one date to the last; the two places that did not were

  * `record_dispersion`, which appended unconditionally. Measured 2026-10-08:
    `improvement/dispersion_history.csv` held **52 rows for 45 distinct dates** -
    2026-04-14 four times, 2026-07-28 three times, 2026-03-16 and 2026-08-26 twice.
    `check_run_health` discards a run whose dispersion is more than 20% below the
    *trailing median* of that file, so a quadruple-counted day moved the bar.
  * `improvement/snapshots/`, which kept every file. Measured the same day: **79 files
    for 52 distinct dates.**

These tests work in a `tmp_path` copy of the module's paths, so they never touch the real
evidence base.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import improvement_engine as ie  # noqa: E402

CATS = ie.CATEGORY_NAMES


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the engine's dispersion file and snapshot dir at a scratch directory."""
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    disp = tmp_path / "dispersion_history.csv"
    monkeypatch.setattr(ie, "SNAPSHOTS_DIR", snaps)
    monkeypatch.setattr(ie, "DISPERSION_HISTORY_PATH", disp)
    return snaps, disp


def _disp(value):
    return {c: value for c in CATS}


# --------------------------------------------------------------------------- dispersion

def test_a_second_run_on_the_same_day_replaces_its_dispersion_row(isolated):
    _, disp = isolated
    ie.record_dispersion("2026-10-08", _disp(10.0))
    ie.record_dispersion("2026-10-08", _disp(12.0))
    out = pd.read_csv(disp)
    assert len(out) == 1, f"expected one row for the date, got {len(out)}"
    assert out.iloc[0][f"{CATS[0]}_disp"] == pytest.approx(12.0), "the later run should win"


def test_different_days_each_keep_a_row(isolated):
    _, disp = isolated
    for d, v in [("2026-10-06", 9.0), ("2026-10-07", 10.0), ("2026-10-08", 11.0)]:
        ie.record_dispersion(d, _disp(v))
    out = pd.read_csv(disp)
    assert len(out) == 3
    assert list(out["date"].astype(str)) == ["2026-10-06", "2026-10-07", "2026-10-08"]


def test_rows_stay_in_date_order_even_when_recorded_out_of_order(isolated):
    """A backfill must not leave the trailing median reading the wrong tail."""
    _, disp = isolated
    for d in ("2026-10-08", "2026-10-02", "2026-10-06"):
        ie.record_dispersion(d, _disp(1.0))
    out = pd.read_csv(disp)
    assert list(out["date"].astype(str)) == ["2026-10-02", "2026-10-06", "2026-10-08"]


def test_the_history_never_gains_more_rows_than_dates(isolated):
    _, disp = isolated
    for d in ("2026-10-05", "2026-10-05", "2026-10-06", "2026-10-06", "2026-10-06"):
        ie.record_dispersion(d, _disp(5.0))
    out = pd.read_csv(disp)
    assert len(out) == out["date"].nunique() == 2


# --------------------------------------------------------------------------- snapshots

def _scored(n=6):
    return pd.DataFrame({
        "Ticker": [f"T{i}" for i in range(n)],
        "Sector": ["Information Technology"] * n,
        "Composite": [50.0 + i for i in range(n)],
        "Rank": list(range(1, n + 1)),
        "_current_price": [100.0 + i for i in range(n)],
        **{c: [50.0 + i for i in range(n)] for c in ie.CATEGORY_SCORES},
    })


def test_a_second_snapshot_for_a_date_supersedes_the_first(isolated):
    snaps, _ = isolated
    ie.record_run_snapshot("runaaaa", "2026-10-08", _scored(), None, {})
    ie.record_run_snapshot("runbbbb", "2026-10-08", _scored(), None, {})
    files = sorted(p.name for p in snaps.glob("*.parquet"))
    assert files == ["2026-10-08_runbbbb.parquet"], files


def test_snapshots_for_other_dates_survive(isolated):
    snaps, _ = isolated
    ie.record_run_snapshot("runaaaa", "2026-10-07", _scored(), None, {})
    ie.record_run_snapshot("runbbbb", "2026-10-08", _scored(), None, {})
    ie.record_run_snapshot("runcccc", "2026-10-08", _scored(), None, {})
    dates = sorted(p.stem.split("_")[0] for p in snaps.glob("*.parquet"))
    assert dates == ["2026-10-07", "2026-10-08"]


def test_the_file_count_equals_the_number_of_distinct_dates(isolated):
    """`generate_improvement_report` counts files, so the count has to mean something."""
    snaps, _ = isolated
    for i, d in enumerate(["2026-10-05", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-07"]):
        ie.record_run_snapshot(f"run{i:04d}", d, _scored(), None, {})
    files = list(snaps.glob("*.parquet"))
    assert len(files) == len({p.stem.split("_")[0] for p in files}) == 3


def test_the_surviving_snapshot_is_the_one_that_was_written_last(isolated):
    snaps, _ = isolated
    ie.record_run_snapshot("runaaaa", "2026-10-08", _scored(), None, {})
    second = _scored()
    second["Composite"] = second["Composite"] + 7.0
    ie.record_run_snapshot("runbbbb", "2026-10-08", second, None, {})
    kept = pd.read_parquet(next(snaps.glob("*.parquet")))
    assert kept["Composite"].min() == pytest.approx(57.0)


# --------------------------------------------------------------------------- the real files

# --------------------------------------------------------------------------- momentum vol

def test_the_momentum_vol_history_keeps_one_row_per_date(tmp_path):
    """`factor_vol_history.csv` is the third file with this defect, found 2026-10-08 when a
    second run that day added a second row for it.

    It matters more than the other two: `adjust_momentum_weight` ranks the current run's
    momentum dispersion against **every row** in this file, and that percentile decides
    whether momentum weight is cut or raised. A date repeated nine times - which 2026-02-21
    was, out of 71 rows - is nine votes for one day's data."""
    import numpy as np
    from factor_engine import adjust_momentum_weight

    cfg = {"factor_weights": {"momentum": 13, "quality": 22, "valuation": 22}}
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"momentum_score": rng.normal(50, 15, 200)})

    for _ in range(3):
        adjust_momentum_weight(df, cfg, root_dir=str(tmp_path))

    hist = pd.read_csv(tmp_path / "factor_vol_history.csv")
    assert len(hist) == 1, f"three runs on one day wrote {len(hist)} rows: {hist.to_dict()}"
    assert hist["date"].astype(str).is_unique


def test_the_momentum_vol_history_keeps_earlier_dates(tmp_path):
    """Replacing today's row must not drop the history the percentile is taken against."""
    import numpy as np
    from factor_engine import adjust_momentum_weight

    path = tmp_path / "factor_vol_history.csv"
    path.write_text("date,momentum_vol\n2026-09-01,10.0\n2026-09-02,11.0\n", encoding="utf-8")
    cfg = {"factor_weights": {"momentum": 13, "quality": 22, "valuation": 22}}
    df = pd.DataFrame({"momentum_score": np.random.default_rng(5).normal(50, 15, 200)})

    adjust_momentum_weight(df, cfg, root_dir=str(tmp_path))
    adjust_momentum_weight(df, cfg, root_dir=str(tmp_path))

    hist = pd.read_csv(path)
    assert list(hist["date"].astype(str)[:2]) == ["2026-09-01", "2026-09-02"]
    assert len(hist) == 3 and hist["date"].astype(str).is_unique


def test_the_committed_evidence_base_has_one_row_and_one_file_per_date():
    """A tripwire on the real files: the repair done on 2026-10-08 must stay repaired.

    If this fails, something has started appending again - find it before reading any
    number off the evidence base."""
    for name in ("improvement/dispersion_history.csv", "factor_vol_history.csv"):
        path = ROOT / name
        if not path.exists():
            continue
        d = pd.read_csv(path)
        dupes = d["date"].astype(str).value_counts()
        dupes = dupes[dupes > 1]
        assert dupes.empty, f"{name} repeats these dates: {dupes.to_dict()}"

    snaps = ROOT / "improvement" / "snapshots"
    if snaps.exists():
        seen = {}
        for p in snaps.glob("*.parquet"):
            seen.setdefault(p.stem.split("_")[0], []).append(p.name)
        dupes = {k: v for k, v in seen.items() if len(v) > 1}
        assert not dupes, f"improvement/snapshots keeps several files per date: {dupes}"
