"""Percentiles are ranked within ``(Sector, size_band)`` - phase 1 of plan/universe-expansion.md.

Pre-registration 3, adopted 2026-11-03: once small companies are in the pool a metric is ranked
within its sector **and** its size band, and a sector-band group below ``SECTOR_MIN_PEERS`` falls
back to its band rather than to the whole universe.

Shipped 2026-11-07. The S&P 500 is one band, so today the new grouping is exactly the sector
grouping it replaces, and that is the phase's exit test. "By construction" is an argument, so
these measure it instead:

* ``test_one_band_reproduces_the_sector_grouping_exactly`` compares the production function with
  a transcription of the sector-only code it replaced, on the published run.
* ``test_a_genuine_two_band_split_moves_the_percentiles`` is its floor. An identity check that
  cannot fail proves nothing - and the pre-registration's own numbers (0 of 23,092 against
  20,151 moved) are only meaningful as a pair.

The band-wide fallback gets its own test, on a frame built so the two fallbacks disagree: with
one band and one sector nothing can tell them apart, which is how a universe-wide fallback could
have shipped under this name.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402
import run_context  # noqa: E402
import universe  # noqa: E402


# --------------------------------------------------------------------------- the definition

def test_the_current_universe_declares_a_band():
    assert universe.CURRENT.size_band == "S&P 500"
    assert universe.SIZE_BAND_COL == "size_band"


def test_assign_fills_a_frame_that_has_no_band():
    df = pd.DataFrame({"Ticker": ["A", "B"]})
    out = universe.assign_size_bands(df)
    assert out is df, "the frame is banded in place; a copy would silently drop the column"
    assert list(df[universe.SIZE_BAND_COL]) == ["S&P 500", "S&P 500"]


def test_assign_keeps_a_band_a_caller_set_and_fills_only_the_blanks():
    """Phase 2 sets three bands itself; a blank is filled because groupby drops a NaN key."""
    df = pd.DataFrame({"Ticker": list("ABCDE"),
                       universe.SIZE_BAND_COL: ["S&P 400", "S&P 600", np.nan, "", "  "]})
    universe.assign_size_bands(df)
    assert list(df[universe.SIZE_BAND_COL]) == [
        "S&P 400", "S&P 600", "S&P 500", "S&P 500", "S&P 500"]


def test_a_named_universe_overrides_the_current_one():
    other = universe.Universe(key="sp400", name="S&P 400", source="t",
                              min_members=390, max_members=410, min_ranked=380,
                              size_band="S&P 400")
    df = pd.DataFrame({"Ticker": ["A"]})
    universe.assign_size_bands(df, other)
    assert list(df[universe.SIZE_BAND_COL]) == ["S&P 400"]


def test_the_scorer_bands_a_frame_that_arrives_without_the_column():
    """`backtest_v2` and the backtests call the scorer directly on frames of their own."""
    df = _toy_frame(n_per_sector=12, sectors=("Tech", "Energy"))
    assert universe.SIZE_BAND_COL not in df.columns
    fe.compute_sector_percentiles(df)
    assert df[universe.SIZE_BAND_COL].notna().all()


# --------------------------------------------------------------------------- the exit test

def _sector_only_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """``compute_sector_percentiles`` as it stood before 2026-11-07, transcribed.

    Group by ``Sector`` alone; below ten valid values fall back to the whole universe. Kept here
    rather than imported so the comparison is of the change and nothing else - if the production
    function is edited this stays put and the test speaks.
    """
    df = df.copy()
    pct = {c: f"{c}_pct" for c in fe.METRIC_COLS}
    for c in pct.values():
        df[c] = np.nan
    universe_ranks = {col: fe._directed_pct(df[col], fe.METRIC_DIR.get(col, True))
                      for col in fe.METRIC_COLS if col in df.columns}
    for _, grp in df.groupby("Sector"):
        for col in fe.METRIC_COLS:
            pc = pct[col]
            if col not in df.columns:
                df.loc[grp.index, pc] = 50.0
                continue
            if len(grp[col].dropna()) < fe.SECTOR_MIN_PEERS:
                ranks = universe_ranks.get(col)
                df.loc[grp.index, pc] = 50.0 if ranks is None else ranks.loc[grp.index]
                continue
            df.loc[grp.index, pc] = fe._directed_pct(grp[col], fe.METRIC_DIR.get(col, True))
    return df


def _count_differences(a: pd.DataFrame, b: pd.DataFrame) -> tuple[int, int]:
    """(values compared, values that differ), NaN == NaN."""
    cols = [f"{c}_pct" for c in fe.METRIC_COLS if f"{c}_pct" in a.columns]
    diffs = 0
    for c in cols:
        x, y = a[c], b[c]
        both_nan = x.isna() & y.isna()
        diffs += int((~both_nan & ~(x.fillna(-1e18) == y.fillna(-1e18))).sum())
    return len(cols) * len(a), diffs


@pytest.fixture(scope="module")
def published_frame() -> pd.DataFrame:
    try:
        run_dir = pathlib.Path(run_context.latest_run_dir())
    except Exception as exc:                                  # pragma: no cover - no runs/ here
        pytest.skip(f"MISSING DATA: no run directory ({exc})")
    path = run_dir / "02_outliers_flagged.parquet"
    if not path.exists():
        pytest.skip(f"MISSING DATA: {run_dir.name} has no 02_outliers_flagged.parquet")
    df = pd.read_parquet(path)
    if len(df) < 100 or "Sector" not in df.columns:
        pytest.skip(f"MISSING DATA: {run_dir.name} is not a scored universe")
    return df


def test_one_band_reproduces_the_sector_grouping_exactly(published_frame):
    """The phase-1 exit test: the S&P 500's percentiles do not move."""
    before = _sector_only_percentiles(published_frame)
    after = fe.compute_sector_percentiles(published_frame.copy())
    compared, diffs = _count_differences(before, after)
    assert compared > 20_000, f"only {compared} values compared; the run looks truncated"
    assert diffs == 0, f"{diffs} of {compared} percentile values moved"


def test_a_genuine_two_band_split_moves_the_percentiles(published_frame):
    """The floor. Without this the test above could be comparing a function with itself."""
    split = published_frame.copy()
    cap = pd.to_numeric(split["size_log_mcap"], errors="coerce")
    split[universe.SIZE_BAND_COL] = np.where(cap >= cap.median(), "big", "small")
    before = _sector_only_percentiles(published_frame)
    after = fe.compute_sector_percentiles(split)
    compared, diffs = _count_differences(before, after)
    assert diffs > compared // 10, (
        f"a half-and-half size split moved only {diffs} of {compared} values; the band is not "
        "reaching the grouping, so the identity test above proves nothing"
    )


@pytest.fixture(scope="module")
def category_frame() -> pd.DataFrame:
    """The run's category scores - ``compute_composite``'s own input, so it can be rerun."""
    try:
        run_dir = pathlib.Path(run_context.latest_run_dir())
    except Exception as exc:                                  # pragma: no cover - no runs/ here
        pytest.skip(f"MISSING DATA: no run directory ({exc})")
    path = run_dir / "04_category_scores.parquet"
    if not path.exists():
        pytest.skip(f"MISSING DATA: {run_dir.name} has no 04_category_scores.parquet")
    df = pd.read_parquet(path)
    if len(df) < 100 or "valuation_score" not in df.columns:
        pytest.skip(f"MISSING DATA: {run_dir.name} is not a scored universe")
    if fe.load_config().get("sector_neutral", {}).get("sector_relative_composite", False):
        pytest.skip("the composite percentile is on its sector-relative branch in this config")
    return df


def test_grouping_the_composite_by_one_band_is_the_ungrouped_rank(category_frame):
    """The change to ``compute_composite``, applied to the run's own composites.

    The published figure cannot be compared with a recomputation of itself: the percentile is
    taken before ``compute_composite`` rounds the composite to 2dp (line ~4059), and 58 of the
    499 stocks on the 2026-11-06 run are tied after that rounding and not before - which is the
    first thing this test was written believing. So the claim pinned here is the one the edit
    rests on, measured on real composites with their real ties: ranking inside a single constant
    group is the ungrouped ranking, exactly.
    """
    out = fe.compute_composite(category_frame.copy(), fe.load_config())
    comp = out["Composite"]
    const = pd.Series(universe.CURRENT.size_band, index=out.index)
    assert ((comp.groupby(const).rank(pct=True) * 100).round(2)
            == (comp.rank(pct=True) * 100).round(2)).all()
    assert int(comp.duplicated().sum()) > 0, (
        "this run has no tied composites, so the comparison above never exercised the tie "
        "handling the two differ on"
    )


def test_the_composite_percentile_would_move_on_a_two_band_split(category_frame):
    """The floor for the test above: the grouping is live, not decorative."""
    df = category_frame.copy()
    cap = pd.to_numeric(df["size_log_mcap"], errors="coerce")
    df[universe.SIZE_BAND_COL] = np.where(cap >= cap.median(), "big", "small")
    out = fe.compute_composite(df, fe.load_config())
    one = (out["Composite"].rank(pct=True) * 100).round(2)
    moved = int((out["Composite_Pct"] != one).sum())
    assert moved > len(out) // 2, (
        f"a half-and-half size split moved only {moved} of {len(out)} composite percentiles; "
        "the band is not reaching compute_composite"
    )


# --------------------------------------------------------------------------- the fallback

#: A metric that points the usual way, so the expected ranks below read in the obvious order.
HIGHER_IS_BETTER = next(c for c in fe.METRIC_COLS if fe.METRIC_DIR.get(c, True) is True)


def _toy_frame(n_per_sector: int, sectors: tuple[str, ...]) -> pd.DataFrame:
    col = HIGHER_IS_BETTER
    rows = []
    for s in sectors:
        for i in range(n_per_sector):
            rows.append({"Ticker": f"{s}{i}", "Sector": s, col: float(i)})
    return pd.DataFrame(rows)


def test_a_thin_sector_band_group_falls_back_to_its_band_not_the_universe():
    """Two bands, and one sector-band cell too thin to rank inside itself.

    The thin cell's values are the *lowest* in their own band and the *highest* overall. Ranked
    band-wide it is near the bottom; ranked universe-wide it is near the top. One band, or one
    sector, and the two answers coincide - which is how a universe-wide fallback could have
    shipped under this name.
    """
    col = HIGHER_IS_BETTER
    rows = []
    # Band "small": 12 Tech peers spanning 0-11, plus 5 thin Utilities at 12-16.
    for i in range(12):
        rows.append({"Ticker": f"T{i}", "Sector": "Tech", "size_band": "small", col: float(i)})
    for i in range(5):
        rows.append({"Ticker": f"U{i}", "Sector": "Utilities", "size_band": "small",
                     col: 12.0 + i})
    # Band "big": 12 Tech peers far above everything in "small".
    for i in range(12):
        rows.append({"Ticker": f"B{i}", "Sector": "Tech", "size_band": "big",
                     col: 100.0 + i})
    df = pd.DataFrame(rows)
    out = fe.compute_sector_percentiles(df)
    thin = out[out["Sector"] == "Utilities"][f"{col}_pct"]

    # Band-wide: the 5 thin rows are the top 5 of 17 in "small" -> midpoint ranks 13..17.
    expected = [(r - 0.5) / 17 * 100 for r in range(13, 18)]
    assert np.allclose(sorted(thin), expected), (
        f"thin-cell percentiles {sorted(thin)} are not the band-wide ranks {expected}"
    )
    # Universe-wide they would be ranks 13..17 of 29 - a different, lower set.
    universe_wide = [(r - 0.5) / 29 * 100 for r in range(13, 18)]
    assert not np.allclose(sorted(thin), universe_wide), "the fallback ranked across both bands"


def test_a_sector_band_cell_at_the_minimum_is_ranked_inside_itself():
    """``SECTOR_MIN_PEERS`` keeps its value; only its scope changed."""
    col = HIGHER_IS_BETTER
    rows = [{"Ticker": f"T{i}", "Sector": "Tech", "size_band": "a", col: float(i)}
            for i in range(fe.SECTOR_MIN_PEERS)]
    rows += [{"Ticker": f"B{i}", "Sector": "Tech", "size_band": "b", col: 100.0 + i}
             for i in range(fe.SECTOR_MIN_PEERS)]
    out = fe.compute_sector_percentiles(pd.DataFrame(rows))
    n = fe.SECTOR_MIN_PEERS
    within = [(r - 0.5) / n * 100 for r in range(1, n + 1)]
    for band in ("a", "b"):
        got = sorted(out[out["size_band"] == band][f"{col}_pct"])
        assert np.allclose(got, within), f"band {band} was not ranked inside itself: {got}"


def test_a_row_with_no_band_is_still_scored():
    """A NaN groupby key drops the row silently; blanks are filled before the grouping."""
    col = HIGHER_IS_BETTER
    df = _toy_frame(n_per_sector=12, sectors=("Tech",))
    df[universe.SIZE_BAND_COL] = [np.nan] * len(df)
    out = fe.compute_sector_percentiles(df)
    assert out[f"{col}_pct"].notna().all(), "a blank band dropped rows out of the scoring"


# --------------------------------------------------------------------------- provenance

def test_universe_is_a_result_path_for_the_research_engine():
    """It decides what each stock is ranked against, so it can move every v2 number."""
    sys.path.insert(0, str(ROOT))
    import v2_experiments as X
    assert "universe.py" in X.RESULT_PATHS
