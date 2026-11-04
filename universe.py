"""The universe being run - one definition every size constant reads (phase 1 of
``plan/universe-expansion.md``, 2026-11-03).

WHY THIS EXISTS. The S&P 500's size was written down as literals in each place that checks it:
``check_run_health.MIN_SCORED = 490``, ``rehearse_publish.MIN_UNIVERSE = 495``,
``universe_history.MIN/MAX_PLAUSIBLE_MEMBERS = 495/515`` and ``>= 490`` floors in tests. Each was
right for the S&P 500, and none of them could become right for the S&P 400 or 600 without every
copy being found by hand. The plan's phase 1 makes them properties of the universe being run; this
module is that, with the S&P 500 the only universe defined and the values unchanged, so nothing
published moves (the phase's exit test).

Still literal, deliberately, and listed so they are not forgotten: ``data-run.ps1``'s
``$universe = 500`` (the data loop's own script, not edited within hours of a run) and the page's
``D.table_data.length || 500`` fallback. ``tests/test_universe_definition.py`` names them.
"""

from __future__ import annotations

from dataclasses import dataclass


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
)

#: The universe this build runs. One today; phase 2 adds the S&P 400 and 600 beside it.
CURRENT = SP500
