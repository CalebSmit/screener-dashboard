"""Tests for history.py - the dashboard's time dimension.

The dashboard had no time dimension at all, so every one of these behaviours is
new. The ones that matter most are the exclusion rules: the snapshot directory
contains a run (``2026-07-28``) whose ranking bears no relation to its
neighbours, and diffing it naively reports 411 of 501 stocks moving more than
50 ranks. A "biggest movers" panel built without these guards would headline
pure artifact, on a tool whose entire product is credibility.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import history as H  # noqa: E402


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

CATS = H.CATEGORIES


def _snap_df(ranks: dict[str, int], cat_overrides: dict | None = None) -> pd.DataFrame:
    """A snapshot frame with the columns record_run_snapshot writes."""
    cat_overrides = cat_overrides or {}
    rows = []
    n = len(ranks)
    for ticker, rank in ranks.items():
        # Composite decreasing in rank keeps the frame internally consistent.
        row = {"Ticker": ticker, "Sector": "Tech", "Rank": rank,
               "Composite": round(100.0 * (n - rank + 1) / n, 2)}
        for cat in CATS:
            row[f"{cat}_score"] = float(cat_overrides.get(ticker, {}).get(cat, 50.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _write(dirpath: Path, date: str, ranks: dict[str, int], run_id="abc123",
           cat_overrides=None) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / f"{date}_{run_id}.parquet"
    _snap_df(ranks, cat_overrides).to_parquet(path)
    return path


def _ranks(order: list[str]) -> dict[str, int]:
    return {t: i + 1 for i, t in enumerate(order)}


UNIVERSE = [f"T{i:03d}" for i in range(60)]

# Materiality defaults to a 55-rank move, so tests about movers need a universe
# big enough for a move that size to exist at all.
BIG = [f"B{i:03d}" for i in range(300)]


def _shift(order: list[str], moves: dict[str, int]) -> list[str]:
    """Reorder by nudging named tickers up (negative) or down (positive).

    Offsets are half-integers so a nudge always crosses a neighbour: with whole
    numbers a +1 shift ties with the next ticker and Python's stable sort
    leaves the order unchanged, silently producing a duplicate run.
    """
    pos = {t: float(i) for i, t in enumerate(order)}
    for t, d in moves.items():
        pos[t] += d + 0.5
    return sorted(order, key=lambda t: pos[t])


def _move_to(order: list[str], ticker: str, new_index: int) -> list[str]:
    rest = [t for t in order if t != ticker]
    rest.insert(new_index, ticker)
    return rest


def _scramble(order: list[str]) -> list[str]:
    """An ordering with essentially no relationship to the input."""
    return order[::3] + order[1::3] + order[2::3]


# ---------------------------------------------------------------------------
# snapshot_index
# ---------------------------------------------------------------------------

def test_snapshot_index_keeps_one_file_per_date(tmp_path):
    """A day with several runs must contribute one point, not several.

    Mirrors improvement_engine.compute_forward_returns: sorted order, last
    file for a date wins. Counting each run file separately is the same
    evidence-inflation failure as CLAUDE.md priority 0.6.
    """
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE), run_id="aaa")
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE), run_id="zzz")
    _write(tmp_path, "2026-05-02", _ranks(UNIVERSE), run_id="bbb")

    index = H.snapshot_index(tmp_path)

    assert sorted(index) == ["2026-05-01", "2026-05-02"]
    assert index["2026-05-01"].name.endswith("zzz.parquet")


def test_snapshot_index_missing_dir_is_empty(tmp_path):
    assert H.snapshot_index(tmp_path / "nope") == {}


# ---------------------------------------------------------------------------
# rank_continuity
# ---------------------------------------------------------------------------

def test_rank_continuity_identical_runs_is_one(tmp_path):
    a = H._to_run_snapshot("2026-05-01", _snap_df(_ranks(UNIVERSE)))
    assert H.rank_continuity(a, a) == pytest.approx(1.0)


def test_rank_continuity_reversed_is_negative(tmp_path):
    a = H._to_run_snapshot("2026-05-01", _snap_df(_ranks(UNIVERSE)))
    b = H._to_run_snapshot("2026-05-02", _snap_df(_ranks(list(reversed(UNIVERSE)))))
    assert H.rank_continuity(a, b) == pytest.approx(-1.0)


def test_rank_continuity_none_when_overlap_too_thin():
    """A correlation over a handful of names is not evidence of anything."""
    a = H._to_run_snapshot("2026-05-01", _snap_df(_ranks(UNIVERSE[:5])))
    b = H._to_run_snapshot("2026-05-02", _snap_df(_ranks(UNIVERSE[:5])))
    assert H.rank_continuity(a, b) is None


# ---------------------------------------------------------------------------
# select_comparable_runs
# ---------------------------------------------------------------------------

def test_degraded_run_is_excluded(tmp_path):
    """The 2026-07-28 failure, reproduced: a run unrelated to its neighbours."""
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-02", _ranks(_shift(UNIVERSE, {"T005": 3})))
    # Scrambled: correlates with nothing.
    scrambled = UNIVERSE[::3] + UNIVERSE[1::3] + UNIVERSE[2::3]
    _write(tmp_path, "2026-05-03", _ranks(scrambled))
    _write(tmp_path, "2026-05-04", _ranks(_shift(UNIVERSE, {"T009": 4})))

    kept, excluded = H.select_comparable_runs(tmp_path)

    assert [s.date for s in kept] == ["2026-05-01", "2026-05-02", "2026-05-04"]
    bad = [e for e in excluded if e["date"] == "2026-05-03"]
    assert len(bad) == 1
    assert bad[0]["reason"] == "rank_discontinuity"


def test_run_after_a_degraded_run_is_compared_to_the_last_good_one(tmp_path):
    """One bad snapshot must not drag the following good run out with it."""
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-02", _ranks(_move_to(UNIVERSE, "T005", 10)))
    _write(tmp_path, "2026-05-03", _ranks(_scramble(UNIVERSE)))
    _write(tmp_path, "2026-05-04", _ranks(_move_to(UNIVERSE, "T009", 20)))

    kept, _ = H.select_comparable_runs(tmp_path)

    assert [s.date for s in kept] == ["2026-05-01", "2026-05-02", "2026-05-04"]


def test_duplicate_run_is_excluded(tmp_path):
    """A re-run off cache is not a new observation (CLAUDE.md priority 0.6)."""
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-02", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-03", _ranks(_shift(UNIVERSE, {"T007": 5})))

    kept, excluded = H.select_comparable_runs(tmp_path)

    assert [s.date for s in kept] == ["2026-05-01", "2026-05-03"]
    dup = [e for e in excluded if e["date"] == "2026-05-02"]
    assert dup and dup[0]["reason"] == "duplicate_of_previous"


def test_degraded_first_run_does_not_reject_the_whole_series(tmp_path):
    """A bad run in first position must not become the reference everything
    else is judged against - that would empty the history."""
    scrambled = UNIVERSE[::3] + UNIVERSE[1::3] + UNIVERSE[2::3]
    _write(tmp_path, "2026-05-01", _ranks(scrambled))
    _write(tmp_path, "2026-05-02", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-03", _ranks(_shift(UNIVERSE, {"T004": 3})))
    _write(tmp_path, "2026-05-04", _ranks(_shift(UNIVERSE, {"T011": 2})))

    kept, excluded = H.select_comparable_runs(tmp_path)

    assert [s.date for s in kept] == ["2026-05-02", "2026-05-03", "2026-05-04"]
    assert excluded[0]["date"] == "2026-05-01"
    assert excluded[0]["reason"] == "rank_discontinuity"


def test_every_exclusion_is_reported_with_a_reason(tmp_path):
    """A silently shortened history is indistinguishable from a stable one."""
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-02", _ranks(UNIVERSE))

    _, excluded = H.select_comparable_runs(tmp_path)

    assert excluded
    for entry in excluded:
        assert entry["date"] and entry["reason"] and entry["detail"]


def test_frame_without_required_columns_is_excluded(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Ticker": ["A"], "Nonsense": [1]}).to_parquet(
        tmp_path / "2026-05-01_x.parquet")
    _write(tmp_path, "2026-05-02", _ranks(UNIVERSE))

    kept, excluded = H.select_comparable_runs(tmp_path)

    assert [s.date for s in kept] == ["2026-05-02"]
    assert excluded[0]["reason"] == "missing_columns"


def test_up_to_date_truncates_rather_than_mixing_eras(tmp_path):
    """Regenerating from an older run must not splice in later snapshots."""
    for d in ["2026-05-01", "2026-05-02", "2026-05-03"]:
        _write(tmp_path, d, _ranks(_shift(UNIVERSE, {"T003": int(d[-1])})))

    kept, _ = H.select_comparable_runs(tmp_path, up_to_date="2026-05-02")

    assert [s.date for s in kept] == ["2026-05-01", "2026-05-02"]


# ---------------------------------------------------------------------------
# rank_change_noise
# ---------------------------------------------------------------------------

def test_noise_falls_back_when_there_are_too_few_pairs(tmp_path):
    kept = [H._to_run_snapshot("2026-05-01", _snap_df(_ranks(UNIVERSE)))]
    noise = H.rank_change_noise(kept)
    assert noise["source"] == "fallback"
    assert noise["material_threshold"] == H.FALLBACK_MATERIAL_RANK_MOVE


def test_noise_ignores_pairs_separated_by_a_long_gap():
    """A multi-month gap would inflate the noise floor and hide real moves."""
    a = H._to_run_snapshot("2026-01-01", _snap_df(_ranks(UNIVERSE)))
    b = H._to_run_snapshot("2026-06-01", _snap_df(_ranks(list(reversed(UNIVERSE)))))
    noise = H.rank_change_noise([a, b])
    assert noise["n_pairs"] == 0
    assert noise["source"] == "fallback"


def test_noise_is_measured_when_enough_close_pairs_exist():
    kept = []
    for i, day in enumerate(["01", "02", "03", "04"]):
        kept.append(H._to_run_snapshot(
            f"2026-05-{day}", _snap_df(_ranks(_shift(UNIVERSE, {"T010": i})))))
    noise = H.rank_change_noise(kept)
    assert noise["source"] == "measured"
    assert noise["n_pairs"] == 3
    assert noise["material_threshold"] >= 1


# ---------------------------------------------------------------------------
# round_trip_tickers
# ---------------------------------------------------------------------------

def test_round_trip_flags_an_excursion_that_returns_to_base():
    """The MNST shape: a metric drops out for two runs, then comes back."""
    paths = [10, 12, 400, 402, 11]
    kept = []
    for i, rank in enumerate(paths):
        order = [t for t in UNIVERSE if t != "T000"]
        order.insert(min(rank, len(order)), "T000")
        kept.append(H._to_run_snapshot(f"2026-05-0{i + 1}", _snap_df(_ranks(order))))

    flagged = H.round_trip_tickers(kept, threshold=20)

    assert "T000" in flagged


def test_round_trip_does_not_flag_a_sustained_trend():
    """A stock that genuinely deteriorated must stay in the movers list."""
    kept = []
    for i, rank in enumerate([2, 12, 22, 34, 48]):
        order = [t for t in UNIVERSE if t != "T000"]
        order.insert(rank, "T000")
        kept.append(H._to_run_snapshot(f"2026-05-0{i + 1}", _snap_df(_ranks(order))))

    flagged = H.round_trip_tickers(kept, threshold=20)

    assert "T000" not in flagged


# ---------------------------------------------------------------------------
# build_history
# ---------------------------------------------------------------------------

def _three_run_dir(tmp_path):
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-04", _ranks(_shift(UNIVERSE, {"T002": 6})))
    _write(tmp_path, "2026-05-05", _ranks(_shift(UNIVERSE, {"T002": 12})))
    return tmp_path


def test_build_history_is_unavailable_with_fewer_than_two_runs(tmp_path):
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE))
    out = H.build_history(snapshots_dir=tmp_path)
    assert out["available"] is False
    assert out["movers"] == {}


def test_build_history_shape(tmp_path):
    out = H.build_history(snapshots_dir=_three_run_dir(tmp_path))

    assert out["available"] is True
    assert out["dates"] == ["2026-05-01", "2026-05-04", "2026-05-05"]
    assert out["current_date"] == "2026-05-05"
    assert out["compare"]["prev"]["date"] == "2026-05-04"
    assert out["compare"]["prev"]["gap_days"] == 1
    for entry in out["series"].values():
        assert len(entry["r"]) == len(out["dates"])
        assert len(entry["c"]) == len(out["dates"])


def test_series_is_restricted_to_the_current_universe(tmp_path):
    """A delisted name has no drill-down, so its series would be dead payload."""
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE + ["GONE"]))
    _write(tmp_path, "2026-05-04", _ranks(_shift(UNIVERSE, {"T002": 6})))

    out = H.build_history(snapshots_dir=tmp_path)

    assert "GONE" not in out["series"]
    assert "T002" in out["series"]


def test_new_ticker_is_marked_rather_than_shown_as_a_mover(tmp_path):
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-04", _ranks(UNIVERSE + ["NEWCO"]))

    out = H.build_history(snapshots_dir=tmp_path)

    assert out["delta"]["NEWCO"]["prev"] == {"new": True}
    assert all(m["t"] != "NEWCO" for m in out["movers"]["prev"]["up"])


def test_rank_delta_sign_means_moved_up_the_table(tmp_path):
    """dr > 0 must mean "improved". Rank 1 is best, so the raw subtraction is
    the other way round and getting it backwards would invert the whole panel.
    """
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE))
    improved = ["T050"] + [t for t in UNIVERSE if t != "T050"]
    _write(tmp_path, "2026-05-04", _ranks(improved))

    out = H.build_history(snapshots_dir=tmp_path)

    assert out["delta"]["T050"]["prev"]["dr"] > 0
    assert out["delta"]["T000"]["prev"]["dr"] < 0


def test_live_run_is_appended_when_not_yet_in_the_snapshot_dir(tmp_path):
    """The dashboard must never be a run behind its own history."""
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-04", _ranks(_shift(UNIVERSE, {"T002": 6})))
    live = _snap_df(_ranks(_shift(UNIVERSE, {"T002": 20})))

    out = H.build_history(current_df=live, current_date="2026-05-05",
                          snapshots_dir=tmp_path)

    assert out["dates"][-1] == "2026-05-05"
    assert out["current_date"] == "2026-05-05"


def test_live_run_is_not_duplicated_when_already_on_disk(tmp_path):
    _three_run_dir(tmp_path)
    live = _snap_df(_ranks(_shift(UNIVERSE, {"T002": 12})))

    out = H.build_history(current_df=live, current_date="2026-05-05",
                          snapshots_dir=tmp_path)

    assert out["dates"].count("2026-05-05") == 1
    assert len(out["dates"]) == 3


def test_movers_report_the_full_count_even_when_the_list_is_capped(tmp_path):
    """Silent truncation reads as "this is everything" when it is not."""
    # Promote 20 names from deep in the table to the top: a lot of material
    # movers, while the run as a whole stays comparable to its predecessor.
    promoted = BIG[250:270]
    reordered = promoted + [t for t in BIG if t not in promoted]
    _write(tmp_path, "2026-05-01", _ranks(BIG))
    _write(tmp_path, "2026-05-04", _ranks(reordered))

    out = H.build_history(snapshots_dir=tmp_path)
    mv = out["movers"]["prev"]

    assert mv["n_up"] == 20
    assert len(mv["up"]) == H.MAX_MOVERS_LISTED
    assert mv["listed"] == H.MAX_MOVERS_LISTED


def test_movers_exclude_moves_below_the_material_threshold(tmp_path):
    """Everyday jitter must not reach the panel - that is what invites churn."""
    _write(tmp_path, "2026-05-01", _ranks(BIG))
    _write(tmp_path, "2026-05-04", _ranks(_move_to(BIG, "B030", 34)))

    out = H.build_history(snapshots_dir=tmp_path)
    threshold = out["noise"]["material_threshold"]
    mv = out["movers"]["prev"]

    assert out["delta"]["B030"]["prev"]["dr"] != 0
    assert (mv["n_up"], mv["n_down"]) == (0, 0)
    for row in mv["up"] + mv["down"]:
        assert abs(row["dr"]) >= threshold


def test_lookback_is_skipped_when_no_run_is_old_enough(tmp_path):
    """Better no second comparison than one against yesterday labelled
    '~1 month'."""
    out = H.build_history(snapshots_dir=_three_run_dir(tmp_path))
    assert out["compare"]["m1"] is None
    assert "m1" not in out["movers"]


def test_lookback_picks_the_run_nearest_a_month_back(tmp_path):
    _write(tmp_path, "2026-04-01", _ranks(UNIVERSE))
    _write(tmp_path, "2026-05-06", _ranks(_shift(UNIVERSE, {"T002": 3})))
    _write(tmp_path, "2026-06-01", _ranks(_shift(UNIVERSE, {"T002": 6})))
    _write(tmp_path, "2026-06-02", _ranks(_shift(UNIVERSE, {"T002": 9})))

    out = H.build_history(snapshots_dir=tmp_path)

    assert out["compare"]["m1"]["date"] == "2026-05-06"
    assert out["compare"]["m1"]["gap_days"] == 27


def test_category_deltas_answer_what_changed(tmp_path):
    _write(tmp_path, "2026-05-01", _ranks(UNIVERSE),
           cat_overrides={"T010": {"quality": 80.0}})
    _write(tmp_path, "2026-05-04", _ranks(_shift(UNIVERSE, {"T010": 30})),
           cat_overrides={"T010": {"quality": 20.0}})

    out = H.build_history(snapshots_dir=tmp_path)

    assert out["delta"]["T010"]["prev"]["cat"]["quality"] == pytest.approx(-60.0)


# ---------------------------------------------------------------------------
# Regression against the real snapshot directory
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not H.SNAPSHOTS_DIR.exists(), reason="no snapshots on disk")
def test_real_snapshots_exclude_the_known_degraded_run():
    """2026-07-28 is the documented failure this module exists to survive.

    Its ranks correlate with the neighbouring run at Spearman 0.016 and a naive
    diff reports 82% of the universe as material movers. If it ever silently
    rejoins the series, the movers panel is publishing fiction.
    """
    index = H.snapshot_index()
    if "2026-07-28" not in index:
        pytest.skip("the degraded snapshot is no longer on disk")

    kept, excluded = H.select_comparable_runs()

    assert "2026-07-28" not in [s.date for s in kept]
    reason = [e["reason"] for e in excluded if e["date"] == "2026-07-28"]
    assert reason == ["rank_discontinuity"]


@pytest.mark.skipif(not H.SNAPSHOTS_DIR.exists(), reason="no snapshots on disk")
def test_real_snapshots_produce_a_usable_history():
    """Guards the cascade bug: an over-eager gate that rejects most real runs
    passes every synthetic test above while making the feature useless."""
    kept, _ = H.select_comparable_runs()
    index = H.snapshot_index()
    if len(index) < 5:
        pytest.skip("too few snapshots to judge")

    assert len(kept) >= 0.7 * len(index)
