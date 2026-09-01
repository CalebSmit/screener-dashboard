"""The metric-coverage drift alert was scoring bank-only metrics against the
whole universe instead of the population they apply to.

`_BANK_ONLY_METRICS` (``pb_ratio``, ``roe``, ``roa``, ``equity_ratio``) are by
design only computed for the ~58 of 502 S&P 500 stocks classified as banks or
financials - a non-bank structurally has no value for them, the same way a
bank-only metric could never be "missing" for a bank. Scoring the drift check
against ``len(df)`` instead of the applicable population made these four read
as ~88% missing on every single run since launch, permanently tripping the
50%-missing "High severity" threshold in `validation/data_quality_log.csv` for
a condition that is not a data-quality problem at all - it is the pipeline
working exactly as designed. Confirmed against the live payload from the
2026-09-01 02:00 run: 58/502 stocks have `pb_ratio` (88.4% missing under the
old, wrong denominator), and the true bank population is 58-59 depending on
the metric.

A permanent false "High severity" alarm is the same failure shape CLAUDE.md's
own watchdog design explicitly guards against elsewhere: an alert that fires
on a condition that isn't wrong trains a reader to stop looking at it, which
is exactly the situation where a real drift would go unnoticed.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_screener import _metric_missing_pct  # noqa: E402


def _frame(n_bank=3, n_nonbank=7):
    """A small mixed universe. Bank-only and non-bank-only metrics are wired
    up exactly as they would be by the real pipeline: present only for the
    population they apply to."""
    n = n_bank + n_nonbank
    is_bank = pd.Series([True] * n_bank + [False] * n_nonbank)
    df = pd.DataFrame({
        "Ticker": [f"T{i}" for i in range(n)],
        "_is_bank_like": is_bank,
        # Bank-only: populated for banks, structurally absent for non-banks.
        "pb_ratio": [1.5 if b else None for b in is_bank],
        # Non-bank-only: populated for non-banks, structurally absent for banks.
        "roic": [None if b else 0.12 for b in is_bank],
        # Universal metric: applies to everyone.
        "beta": [1.0] * n,
    })
    return df, is_bank


def test_bank_only_metric_scores_against_banks_not_the_universe():
    """This is the exact bug: previously `pb_ratio` present for every bank and
    absent for every non-bank still read as (n_nonbank / n) missing, when the
    true answer - scored against the population it applies to - is 0%."""
    df, is_bank = _frame(n_bank=3, n_nonbank=7)
    assert _metric_missing_pct(df, "pb_ratio", is_bank) == 0.0


def test_nonbank_only_metric_scores_against_nonbanks_not_the_universe():
    df, is_bank = _frame(n_bank=3, n_nonbank=7)
    assert _metric_missing_pct(df, "roic", is_bank) == 0.0


def test_a_genuinely_missing_bank_metric_is_still_caught():
    """The fix must not silence a real defect - if pb_ratio is actually
    missing for half the banks, that must still show up as 50%, not 0%."""
    df, is_bank = _frame(n_bank=4, n_nonbank=6)
    df.loc[[0, 1], "pb_ratio"] = None  # 2 of 4 real banks now missing it
    assert _metric_missing_pct(df, "pb_ratio", is_bank) == 50.0


def test_universal_metric_still_scores_against_the_whole_universe():
    """Metrics with no population restriction are unaffected by the fix."""
    df, is_bank = _frame(n_bank=3, n_nonbank=7)
    df.loc[0, "beta"] = None
    assert _metric_missing_pct(df, "beta", is_bank) == 10.0


def test_missing_column_is_100pct_not_a_crash():
    df, is_bank = _frame()
    assert _metric_missing_pct(df, "does_not_exist", is_bank) == 100.0


def test_matches_the_real_payload_88pct_figure_under_the_old_denominator():
    """Reproduces the exact false-alarm number from the live
    validation/data_quality_log.csv entry, to pin what "wrong" looked like."""
    n_bank, n_total = 58, 502
    is_bank = pd.Series([True] * n_bank + [False] * (n_total - n_bank))
    df = pd.DataFrame({"pb_ratio": [1.0] * n_bank + [None] * (n_total - n_bank)},
                       index=range(n_total))
    old_wrong_pct = df["pb_ratio"].isna().sum() / len(df) * 100
    assert round(old_wrong_pct, 1) == 88.4, "the old (buggy) figure this fix corrects"
    assert _metric_missing_pct(df, "pb_ratio", is_bank) == 0.0


@pytest.mark.parametrize("n_bank", [0])
def test_empty_applicable_population_is_zero_not_a_zero_division(n_bank):
    """No banks in the universe at all - denom is 0, must not raise."""
    df, is_bank = _frame(n_bank=n_bank, n_nonbank=5)
    assert _metric_missing_pct(df, "pb_ratio", is_bank) == 0.0
