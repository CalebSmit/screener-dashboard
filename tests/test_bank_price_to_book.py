"""The bank price-to-book is market value over common equity, with no per-share figure in it.

Changed 2026-10-20. Proposal: `research/2026-10-19-bank-price-to-book.md` section 8, raised by the
2026-10-19 session and reviewed and re-measured by this one
(`research/measurements/2026-10-20-bank-pb-review.py`). P/B carries 60 of the bank Valuation
table's 100 points - about 13.2 composite points for each of the 46 bank-like stocks - and it used
to read the data source's `.info` `priceToBook`, which is `currentPrice / bookValue` with
`bookValue` a **per-share** figure.

What that cost, measured on the published 2026-10-20 run: Berkshire Hathaway's `bookValue` is book
per **Class A** share ($522,226) against a Class B price (about $532), so its P/B published
**0.0010** where its own filed equity over its market value is **1.5234** - the cheapest bank-like
stock by a factor of a thousand, at the 98.9th percentile of the bank P/B ranking in all 52
improvement snapshots since March. A company-level ratio has no per-share unit to get wrong, and it
sits on the same balance sheet as the rest of the record (the source's book was a quarter newer
than every other figure scored for 18 of the 46).

These tests hold the construction, the absence of a fallback (the Phase 13 F19 rule: one
definition for every stock in the population, never a second one for the stocks the first cannot
reach), and the sentence the public methodology page makes of it (claim
`overview.bank_price_to_book`).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402
import metric_lineage as ml  # noqa: E402

MR = pd.Series(dtype=float)

# A bank-like record carrying only what P/B needs. `_is_bank_like` puts Diversified Banks on the
# bank metric set (tests/test_bank_like.py holds that rule).
BANK = {"Ticker": "TESTBANK", "sector": "Financials", "industry": "Banks - Diversified",
        "_gics_sub": "Diversified Banks"}


def _run(**fields):
    return fe.compute_metrics([BANK | fields], MR, cfg={}).iloc[0]


def test_the_metric_is_market_value_over_common_equity():
    row = _run(marketCap=2.0e11, commonEquity=1.0e11)
    assert row["_is_bank_like"] is True or bool(row["_is_bank_like"])
    assert row["pb_ratio"] == pytest.approx(2.0)


def test_there_is_no_fallback_to_the_sources_ratio():
    """A missing common equity costs the metric; it does not reach for another definition."""
    row = _run(marketCap=2.0e11, priceToBook=1.37, bookValue=40.0, currentPrice=54.8)
    assert pd.isna(row["pb_ratio"]), "the source's priceToBook must not reach the metric"

    row = _run(marketCap=2.0e11, commonEquity=np.nan, priceToBook=1.37)
    assert pd.isna(row["pb_ratio"])
    # ... and the source's ratio does not override a computed one either.
    row = _run(marketCap=2.0e11, commonEquity=1.0e11, priceToBook=1.37, bookValue=40.0)
    assert row["pb_ratio"] == pytest.approx(2.0)


@pytest.mark.parametrize("mc,ce", [(2.0e11, 0.0), (2.0e11, -5.0e10), (0.0, 1.0e11),
                                   (np.nan, 1.0e11), (2.0e11, np.nan)])
def test_a_non_positive_or_missing_input_gives_no_ratio(mc, ce):
    assert pd.isna(_run(marketCap=mc, commonEquity=ce)["pb_ratio"])


def test_a_class_a_book_value_per_share_cannot_reach_the_metric():
    """Berkshire's defect, as its own figures: the published P/B was 1,495x too low.

    Market value and the Class B price are on the B unit; `bookValue` and the balance sheet's
    share count are on the A unit, 1,500 B shares to one A share. Any construction that divides a
    B price by an A book value is wrong by that factor, and the new one never sees a share count.
    """
    mc, equity = 1_139_349_913_600.0, 747_910_000_000.0      # the 2026-10-20 run's own figures
    row = _run(marketCap=mc, commonEquity=equity, priceToBook=0.0010191567,
               bookValue=522_225.90, currentPrice=532.26)
    assert row["pb_ratio"] == pytest.approx(mc / equity, rel=1e-9)
    assert row["pb_ratio"] == pytest.approx(1.5234, abs=5e-4)
    assert row["pb_ratio"] / 0.0010191567 == pytest.approx(1494.8, rel=1e-3)


def test_a_non_bank_is_still_unscored_on_it():
    row = fe.compute_metrics(
        [{"Ticker": "AAPL", "sector": "Information Technology", "industry": "Consumer Electronics",
          "_gics_sub": "Technology Hardware, Storage & Peripherals",
          "marketCap": 2.0e11, "commonEquity": 1.0e11}], MR, cfg={}).iloc[0]
    assert pd.isna(row["pb_ratio"])


def test_the_fetch_reads_common_equity_from_the_balance_sheet_at_the_statement_date():
    """One balance-sheet column for the whole record: the same one `totalEquity` is read from."""
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    assert re.search(r'rec\["commonEquity"\]\s*=\s*_stmt_val\(_bs_src, "Common Stock Equity"\)', src)
    # no column argument -> position 0 of _populated_columns, as for Stockholders Equity
    assert not re.search(r'rec\["commonEquity"\].*_bs_prior_col', src)


def test_the_lineage_the_drilldown_shows_rebuilds_the_metric():
    """Row 0.8c: the equation on the row is the arithmetic the engine did."""
    inputs = {"marketCap": 1_139_349_913_600.0, "commonEquity": 747_910_000_000.0}
    assert ml.RECOMPUTE["pb_ratio"](inputs) == pytest.approx(1.5234, abs=5e-4)
    assert ml.RECOMPUTE["pb_ratio"]({"marketCap": 2.0e11}) is None
    assert ml.RECOMPUTE["pb_ratio"]({"marketCap": 2.0e11, "commonEquity": 0.0}) is None
    # the source's per-share fields are no longer inputs to it
    exact, templates = ml.EQUATIONS["pb_ratio"]
    assert exact is True and len(templates) == 1
    assert "{marketCap|" in templates[0] and "{commonEquity|" in templates[0]
    assert "priceToBook" not in templates[0] and "bookValue" not in templates[0]
    named = {k for k, _, _ in (tuple(x) for x in ml.LINEAGE["pb_ratio"]["inputs"])}
    assert named == {"marketCap", "commonEquity"} or \
        {i[1] for i in ml.LINEAGE["pb_ratio"]["inputs"]} == {"marketCap", "commonEquity"}


def test_the_page_states_the_construction_the_engine_uses():
    """Read the generator's output, never the committed markdown (rule 10)."""
    import yaml

    import run_screener as rs
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    page = rs.build_screener_overview(cfg)
    # the generator's typography pass turns " - " into a spaced en dash
    assert re.search(r"Market value divided by common shareholders' equity [–—] stockholders' "
                     r"equity less preferred stock", page), \
        "the overview must say how bank P/B is computed"
    # the metric table's own row, which used to describe a per-share ratio
    assert "Market value divided by common shareholders' equity at the balance-sheet date" in page
    assert "Share price divided by book value per share" not in page
    # and it says there is no second definition for the stocks the first cannot reach
    assert "scores no P/B rather than one on a second definition" in page


def test_a_point_in_time_record_carries_the_field_it_now_needs():
    """Without this, backtest v2's bank P/B would be NaN for every bank-like stock."""
    import pit_record as pr
    classified = {f for fields in pr.FIELD_CLASSES.values() for f in fields}
    assert "commonEquity" in classified
    assert "commonEquity" in pr.FIELD_CLASSES["statement"]
    src = (ROOT / "pit_record.py").read_text(encoding="utf-8")
    assert re.search(r'r\["commonEquity"\]\s*=\s*r\["totalEquity"\]', src)
