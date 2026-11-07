"""The universe being run - one definition every size constant reads (phase 1 of
``plan/universe-expansion.md``, 2026-11-03).

WHY THIS EXISTS. The S&P 500's size was written down as literals in each place that checks it:
``check_run_health.MIN_SCORED = 490``, ``rehearse_publish.MIN_UNIVERSE = 495``,
``universe_history.MIN/MAX_PLAUSIBLE_MEMBERS = 495/515`` and ``>= 490`` floors in tests. Each was
right for the S&P 500, and none of them could become right for the S&P 400 or 600 without every
copy being found by hand. The plan's phase 1 makes them properties of the universe being run; this
module is that, with the S&P 500 the only universe defined and the values unchanged, so nothing
published moves (the phase's exit test).

Still literal, deliberately, and listed so it is not forgotten: ``data-run.ps1``'s
``$universe = 500`` (the data loop's own script, not edited within hours of a run).
``tests/test_universe_definition.py`` names it. (The page's ``|| 500`` fallback went the same night.)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Universe:
    key: str
    name: str
    #: Where the constituent list comes from, as ``universe_history`` reads it.
    source: str
    #: The plausible constituent count, as an index list parses (share classes included). Outside
    #: it the list is refused - the wrong table, a truncated fetch, a vandalised page.
    min_members: int
    max_members: int
    #: The fewest companies a healthy run ranks: the data loop's health gate fails below it.
    min_ranked: int
    #: The size band its members are ranked inside, beside their sector (phase 1,
    #: pre-registration 3 of ``plan/universe-expansion.md``). One band per S&P index in phases
    #: 2-3; in phase 4 the band is a property of the *stock* - NYSE market-cap breakpoints - and
    #: ``assign_size_bands`` below is where that arrives.
    size_band: str


# Measured, not guessed (the reasoning moved here from the three modules that held it):
# - members 495-515: every revision sampled across the backtest window (2020-01-31 to 2026-09-23)
#   parsed to 503, 505 or 504 rows; widened so share-class churn cannot trip it
#   (universe_history, 2026-09-24).
# - ranked >= 490: 498-503 on every published run, 500 companies after share classes
#   (check_run_health, the trading bot's input gates, 2026-10-22).
SP500 = Universe(
    key="sp500",
    name="S&P 500",
    source="Wikipedia, List of S&P 500 companies (current page and its revision history)",
    min_members=495,
    max_members=515,
    min_ranked=490,
    size_band="S&P 500",
)

#: The universe this build runs. One today; phase 2 adds the S&P 400 and 600 beside it.
CURRENT = SP500

#: The column every frame carries from the fetch onward. Named once so the grouping, the
#: pipeline and the tests cannot disagree about its spelling.
SIZE_BAND_COL = "size_band"


def assign_size_bands(df: pd.DataFrame, universe: Universe | None = None) -> pd.DataFrame:
    """Give every row a size band, in place, and return the frame.

    Today there is one universe and therefore one band, so this writes one constant string and
    changes no number anywhere - which is phase 1's exit test, not an accident. What it buys is
    that the *grouping* below it is already the banded one: when phase 2 scores the S&P 400 and
    600 beside the 500, the only thing that has to change is this function.

    A band already set is never overwritten, so a caller (phase 2, or a test that needs two
    bands) can set its own. Blanks are filled rather than left: ``groupby`` drops a row whose
    key is NaN, and a stock silently dropped out of its own sector's distribution is exactly
    the failure the S&P 500's byte-identity is supposed to catch.
    """
    u = universe or CURRENT
    if SIZE_BAND_COL not in df.columns:
        df[SIZE_BAND_COL] = u.size_band
        return df
    col = df[SIZE_BAND_COL]
    blank = col.isna() | (col.astype(str).str.strip().isin(("", "nan", "None")))
    if blank.any():
        # Rebuilt rather than assigned in place: an all-NaN column is float64, and filling it
        # with a string row by row is deprecated in pandas and silently object-casts today.
        df[SIZE_BAND_COL] = col.astype(object).where(~blank, u.size_band)
    return df
