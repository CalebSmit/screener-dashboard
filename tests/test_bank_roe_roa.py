"""Bank ROE and ROA are net income over AVERAGE balances, with no reach for the source's ratios.

Changed 2026-10-20. Proposal: `research/2026-10-19-bank-roe-roa.md` section 7, raised by the
2026-10-19 session and reviewed and re-measured by this one
(`research/measurements/2026-10-20-bank-roe-roa-review.py`). ROE and ROA carry 35 and 25 of the
bank Quality table's 100 points - about 13.2 composite points for each of the 46 bank-like
stocks - and they used to read the data source's `.info` `returnOnEquity` / `returnOnAssets`:
two ratios with no published definition, which is the objection that removed `earningsGrowth`
from the engine as "a black-box input with no verifiable definition".

What the review measured, on the 2026-10-20 run and a fresh probe:

- **Availability, the refusal test: 46 of 46.** Net income and both balance sheets present and
  positive for every bank-like stock, so the change creates no cohort scored on something else.
  (This is the test the earnings-definition proposal failed on 2026-10-16, with 23 companies
  carrying no figure at all.) Berkshire is included because `_year_ago_position` finds its
  year-ago balance sheet at column **3**, not column four.
- **Against the companies' own SEC filings:** the construction below agrees within 0.5% for
  **37 of 37** comparable stocks (ROA 36 of 37, worst 0.79%); the source's ratios agree within
  2% for **21** (ROE) and **12** (ROA) of 35, because they are built on consolidated income and
  equity including the minority holders' share of partly-owned subsidiaries.
- **It removes a mixed basis rather than creating one.** The source serves no ratio for MS or
  WRB, so those two were scored on net income over *ending* balances inside the same percentile
  ranking as 44 stocks on the source's construction - the Phase 13 F19 defect.

These tests hold the construction, the absence of any fallback, the equation the drilldown
shows, the single shared ROE definition, and the sentence the public methodology page makes of
it (claim `overview.bank_roe_roa`).
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

BANK = {"Ticker": "TESTBANK", "sector": "Financials", "industry": "Banks - Diversified",
        "_gics_sub": "Diversified Banks"}


def _run(**fields):
    return fe.compute_metrics([BANK | fields], MR, cfg={}).iloc[0]


def test_roe_is_net_income_over_average_equity():
    row = _run(netIncome=1.5e10, totalEquity=1.1e11, totalEquity_prior=0.9e11)
    assert bool(row["_is_bank_like"])
    assert row["roe"] == pytest.approx(1.5e10 / 1.0e11)        # average of 110bn and 90bn
    # and it is NOT the ending-balance ratio, which would be 0.1364
    assert row["roe"] != pytest.approx(1.5e10 / 1.1e11)


def test_roa_is_net_income_over_average_assets():
    row = _run(netIncome=1.5e10, totalAssets=2.1e12, totalAssets_prior=1.9e12)
    assert row["roa"] == pytest.approx(1.5e10 / 2.0e12)
    assert row["roa"] != pytest.approx(1.5e10 / 2.1e12)


def test_there_is_no_fallback_to_the_sources_ratios():
    """A missing prior balance costs the metric; it does not reach for another definition.

    Both halves matter. The source's ratio must not fill a gap (that is the mixed-basis defect
    MS and WRB were living in), and it must not override a computed value either.
    """
    row = _run(netIncome=1.5e10, totalEquity=1.1e11, totalAssets=2.1e12,
               returnOnEquity=0.1364, returnOnAssets=0.00714)
    assert pd.isna(row["roe"]), "the source's returnOnEquity must not reach the metric"
    assert pd.isna(row["roa"]), "the source's returnOnAssets must not reach the metric"

    # nor as an override of the computed figure
    row = _run(netIncome=1.5e10, totalEquity=1.1e11, totalEquity_prior=0.9e11,
               totalAssets=2.1e12, totalAssets_prior=1.9e12,
               returnOnEquity=0.99, returnOnAssets=0.99)
    assert row["roe"] == pytest.approx(0.15)
    assert row["roa"] == pytest.approx(0.0075)

    # and the engine no longer reads either field at all
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    body = src[src.index("def compute_metrics"):]
    for f in ("returnOnEquity", "returnOnAssets"):
        assert f'd.get("{f}"' not in body, f"compute_metrics must not read {f}"


@pytest.mark.parametrize("eq,prior", [(0.0, 0.9e11), (1.1e11, 0.0), (-1.0e10, 0.9e11),
                                      (1.1e11, -1.0e10), (np.nan, 0.9e11), (1.1e11, np.nan)])
def test_a_non_positive_or_missing_balance_gives_no_roe(eq, prior):
    assert pd.isna(_run(netIncome=1.5e10, totalEquity=eq, totalEquity_prior=prior)["roe"])


@pytest.mark.parametrize("ta,prior", [(0.0, 1.9e12), (2.1e12, 0.0), (-1.0e11, 1.9e12),
                                      (2.1e12, -1.0e11), (np.nan, 1.9e12), (2.1e12, np.nan)])
def test_a_non_positive_or_missing_balance_gives_no_roa(ta, prior):
    assert pd.isna(_run(netIncome=1.5e10, totalAssets=ta, totalAssets_prior=prior)["roa"])


def test_a_loss_making_bank_still_scores_a_negative_return():
    """Net income may be negative - only the balances are required to be positive."""
    row = _run(netIncome=-2.0e9, totalEquity=1.1e11, totalEquity_prior=0.9e11,
               totalAssets=2.1e12, totalAssets_prior=1.9e12)
    assert row["roe"] == pytest.approx(-0.02)
    assert row["roa"] == pytest.approx(-1.0e-3)


def test_a_non_bank_is_still_unscored_on_them():
    row = fe.compute_metrics(
        [{"Ticker": "AAPL", "sector": "Information Technology", "industry": "Consumer Electronics",
          "_gics_sub": "Technology Hardware, Storage & Peripherals",
          "netIncome": 1.0e11, "totalEquity": 6.0e10, "totalEquity_prior": 5.0e10,
          "totalAssets": 3.5e11, "totalAssets_prior": 3.4e11}], MR, cfg={}).iloc[0]
    assert pd.isna(row["roe"]) and pd.isna(row["roa"])


def test_the_year_ago_balance_is_matched_by_date_not_by_column_position():
    """Berkshire's case: its quarterly balance sheet omits a quarter, so column four is fifteen
    months back. `_year_ago_position` is what makes 46 of 46 computable."""
    cols = ["2026-06-30", "2026-03-31", "2025-12-31", "2025-06-30", "2025-03-31"]
    j = fe._year_ago_position([pd.Timestamp(c) for c in cols],
                              tol=fe.YEAR_AGO_TOL_BALANCE_DAYS)
    assert j == 3, "the 2025-06-30 column, not column four"
    # and the fetch asks for the prior balances at that column
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    assert re.search(r'_bs_prior_col\s*=\s*_year_ago_position\(\s*_bs_cols,\s*'
                     r'tol=YEAR_AGO_TOL_BALANCE_DAYS\s*\)', src)
    for f in ("totalEquity_prior", "totalAssets_prior"):
        assert re.search(rf'rec\["{f}"\].*_bs_prior_col', src), \
            f"{f} must be read at the date-matched column"


def test_the_lineage_the_drilldown_shows_rebuilds_the_metrics():
    """Row 0.8c: the equation on the row is the arithmetic the engine did."""
    eq = {"netIncome": 1.5e10, "totalEquity": 1.1e11, "totalEquity_prior": 0.9e11}
    ta = {"netIncome": 1.5e10, "totalAssets": 2.1e12, "totalAssets_prior": 1.9e12}
    assert ml.RECOMPUTE["roe"](eq) == pytest.approx(0.15)
    assert ml.RECOMPUTE["roa"](ta) == pytest.approx(0.0075)
    # a missing or non-positive prior balance yields nothing, as in the engine
    assert ml.RECOMPUTE["roe"]({"netIncome": 1.5e10, "totalEquity": 1.1e11}) is None
    assert ml.RECOMPUTE["roa"]({"netIncome": 1.5e10, "totalAssets": 2.1e12}) is None
    assert ml.RECOMPUTE["roe"](eq | {"totalEquity_prior": 0.0}) is None
    assert ml.RECOMPUTE["roa"](ta | {"totalAssets_prior": -1.0}) is None
    # the source's ratios are no longer inputs, and each metric has one exact equation
    for metric, fields in (("roe", {"netIncome", "totalEquity", "totalEquity_prior"}),
                           ("roa", {"netIncome", "totalAssets", "totalAssets_prior"})):
        exact, templates = ml.EQUATIONS[metric]
        assert exact is True and len(templates) == 1, f"{metric} needs one exact equation"
        for f in fields:
            assert f"{{{f}|" in templates[0], f"{metric}'s equation must show {f}"
        assert "returnOn" not in templates[0]
        assert {i[1] for i in ml.LINEAGE[metric]["inputs"]} == fields
        assert "Yahoo" not in (ml.LINEAGE[metric].get("how") or "")


def test_sustainable_growth_uses_the_same_roe():
    """There is now one ROE, not two - the caveat that warned readers otherwise is gone."""
    caveat = ml.LINEAGE["sustainable_growth"].get("caveat") or ""
    assert "not the Yahoo ROE" not in caveat
    assert "same basis" in caveat
    # both are net income over average equity, so they agree on the same inputs
    inputs = {"netIncome": 1.5e10, "totalEquity": 1.1e11, "totalEquity_prior": 0.9e11}
    row = _run(**inputs, dividendsPaid=-3.0e9)
    assert row["roe"] == pytest.approx(ml.RECOMPUTE["roe"](inputs))
    # sustainable growth = that ROE x retention (1 - 3/15)
    assert row["sustainable_growth"] == pytest.approx(0.15 * 0.8, rel=1e-6)


def test_the_page_states_the_construction_the_engine_uses():
    """Read the generator's output, never the committed markdown (rule 10)."""
    import yaml

    import run_screener as rs
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    page = rs.build_screener_overview(cfg)
    # the generator's typography pass turns " - " into a spaced en dash
    assert re.search(r"divided by the average of shareholders' equity [–—] or of total assets "
                     r"[–—] at the latest balance-sheet date and a year earlier", page), \
        "the overview must say how bank ROE and ROA are computed"
    # it names the sources for averaging rather than asserting it
    assert "Quarterly Banking Profile" in page and "UBPR" in page
    # it says what it replaced and why, with the measurement
    assert "no published definition anywhere" in page
    assert "37 of 37" in page
    # ... and the honest asymmetry in ROA's denominator
    assert "how productively the group uses them" in page
    # the metric table's own rows
    assert "divided by average shareholders' equity" in page
    assert "divided by average total assets" in page


def test_a_point_in_time_record_no_longer_carries_the_sources_ratios():
    """They were a third construction (ending balances). The inputs the metrics now use are
    statement fields, which `statement_fields` restates as of the month being scored."""
    import pit_record as pr
    classified = {f for fields in pr.FIELD_CLASSES.values() for f in fields}
    for f in ("returnOnEquity", "returnOnAssets"):
        assert f not in classified, f"{f} is no longer read by compute_metrics"
    for f in ("netIncome", "totalEquity", "totalEquity_prior",
              "totalAssets", "totalAssets_prior"):
        assert f in pr.FIELD_CLASSES["statement"], f"{f} must be a point-in-time statement field"
    src = (ROOT / "pit_record.py").read_text(encoding="utf-8")
    assert 'out["returnOnEquity"]' not in src and 'out["returnOnAssets"]' not in src
