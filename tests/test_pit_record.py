"""The engine's raw record rebuilt from SEC filings as they stood on a date (``pit_record.py``).

Backtest v2 scores history with the engine's own ``compute_metrics`` (plan/backtest-v2.md step 3),
so what is pinned here is the record: a figure filed after the date is invisible, a restatement
filed before it wins, trailing twelve months are built the way the filings allow, a field the
filings cannot supply is NaN rather than today's value, every raw field the engine reads has a
declared class, and nothing in production or ``backtest.py`` v1 imports the module.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pit_record as pr  # noqa: E402

NI = "NetIncomeLoss"


def _f(rows):
    return pd.DataFrame(
        [{"ticker": "AAA", "concept": c, "start": pd.Timestamp(s) if s else pd.NaT,
          "end": pd.Timestamp(e), "filed": pd.Timestamp(fd), "form": form, "fp": None, "val": v}
         for c, s, e, fd, form, v in rows])


# A calendar-year filer. FY2024 = 400; 2025 quarters 110, 120, 130 (as year-to-date figures);
# 2024 quarters 90, 100, 105 (comparatives in the 2025 10-Qs).
ROWS = [
    (NI, "2024-01-01", "2024-12-31", "2025-02-10", "10-K", 400.0),
    (NI, "2025-01-01", "2025-03-31", "2025-05-05", "10-Q", 110.0),
    (NI, "2024-01-01", "2024-03-31", "2025-05-05", "10-Q", 90.0),
    (NI, "2025-01-01", "2025-06-30", "2025-08-04", "10-Q", 230.0),
    (NI, "2024-01-01", "2024-06-30", "2025-08-04", "10-Q", 190.0),
    (NI, "2025-01-01", "2025-09-30", "2025-11-03", "10-Q", 360.0),
    (NI, "2024-01-01", "2024-09-30", "2025-11-03", "10-Q", 295.0),
    ("Assets", None, "2025-09-30", "2025-11-03", "10-Q", 5000.0),
    ("Assets", None, "2024-09-30", "2024-11-04", "10-Q", 4000.0),
]


def test_ttm_is_ytd_plus_prior_year_less_prior_ytd():
    F = pr.Filed(_f(ROWS), "AAA", "2025-11-10")
    assert F.ttm("net_income", pd.Timestamp("2025-09-30")) == pytest.approx(360 + 400 - 295)
    assert F.ttm("net_income", pd.Timestamp("2024-12-31")) == 400.0          # a fiscal year ends there


def test_a_single_quarter_is_the_difference_of_two_year_to_dates():
    F = pr.Filed(_f(ROWS), "AAA", "2025-11-10")
    assert F.quarter("net_income", pd.Timestamp("2025-09-30")) == pytest.approx(130.0)
    assert F.quarter("net_income", pd.Timestamp("2025-03-31")) == pytest.approx(110.0)
    # Q4 from the fiscal year less nine months
    assert F.quarter("net_income", pd.Timestamp("2024-12-31")) == pytest.approx(400 - 295)


def test_nothing_filed_after_the_date_is_visible():
    F = pr.Filed(_f(ROWS), "AAA", "2025-11-02")        # the day before the Q3 10-Q
    assert F.latest_end(("net_income",), 80, 380) == pd.Timestamp("2025-06-30")
    assert np.isnan(F.instant("assets", pd.Timestamp("2025-09-30")))
    s = pr.statement_fields(_f(ROWS), "AAA", "2025-11-02")
    assert s["_stmt_date_financials"] == "2025-06-30"
    assert s["netIncome"] == pytest.approx(230 + 400 - 190)


def test_a_restatement_filed_before_the_date_wins_and_one_filed_after_does_not():
    rows = ROWS + [(NI, "2024-01-01", "2024-12-31", "2025-06-01", "10-K/A", 380.0)]
    assert pr.Filed(_f(rows), "AAA", "2025-05-31").fy("net_income", pd.Timestamp("2024-12-31")) == 400.0
    assert pr.Filed(_f(rows), "AAA", "2025-06-01").fy("net_income", pd.Timestamp("2024-12-31")) == 380.0


def test_the_record_keeps_no_value_the_filings_cannot_supply():
    base = {"Ticker": "AAA", "totalRevenue": 9e9, "capex": -1e9, "marketCap": 1e12,
            "_beneish_sga": 5.0, "price_latest": 10.0, "forwardEps": 2.0, "currentPrice": 10.0}
    rec = pr.record_as_of(base, _f(ROWS), "2025-11-10", price=10.0)
    assert np.isnan(rec["totalRevenue"]) and np.isnan(rec["capex"]) and np.isnan(rec["_beneish_sga"])
    assert np.isnan(rec["marketCap"])                     # no share count filed
    assert rec["netIncome"] == pytest.approx(465.0)
    assert rec["totalAssets"] == 5000.0 and rec["totalAssets_prior"] == 4000.0
    # price-history and estimate fields are the caller's, untouched
    assert rec["price_latest"] == 10.0 and rec["forwardEps"] == 2.0


def test_market_fields_divide_the_filed_figures_by_the_given_price():
    rows = ROWS + [("CommonStockSharesOutstanding", None, "2025-09-30", "2025-11-03", "10-Q", 100.0),
                   ("StockholdersEquity", None, "2025-09-30", "2025-11-03", "10-Q", 2000.0),
                   ("LongTermDebtNoncurrent", None, "2025-09-30", "2025-11-03", "10-Q", 300.0),
                   ("CashAndCashEquivalentsAtCarryingValue", None, "2025-09-30", "2025-11-03", "10-Q", 50.0)]
    rec = pr.record_as_of({"Ticker": "AAA"}, _f(rows), "2025-11-10", price=25.0)
    assert rec["marketCap"] == 2500.0
    assert rec["enterpriseValue"] == 2500.0 + 300.0 - 50.0
    assert rec["bookValue"] == 20.0 and rec["priceToBook"] == pytest.approx(1.25)
    assert rec["returnOnEquity"] == pytest.approx(465.0 / 2000.0)


# --------------------------------------------------------------------------- every field classified
def _fields_the_engine_reads() -> set[str]:
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    bodies = [ast.get_source_segment(src, n) for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in ("compute_metrics", "_beneish_parts",
                                                              "_compute_beneish_mscore")]
    text = "\n".join(bodies)
    found = set(re.findall(r'(?:_safe\(d,\s*|d\.get\(|d\[)"([A-Za-z0-9_]+)"', text))
    for args in re.findall(r"_coalesce\(d,([^)]*)\)", text):
        found |= set(re.findall(r'"([A-Za-z0-9_]+)"', args))
    return found


def test_every_raw_field_the_engine_reads_has_a_class():
    """A field nobody classified would keep today's value in every historical month."""
    read = _fields_the_engine_reads()
    assert len(read) > 80                                      # the scan is not vacuous
    classified = {f for fields in pr.FIELD_CLASSES.values() for f in fields}
    assert read - classified == set(), sorted(read - classified)


def test_the_classes_do_not_overlap():
    seen = {}
    for cls, fields in pr.FIELD_CLASSES.items():
        for f in fields:
            assert f not in seen, f"{f} in {seen.get(f)} and {cls}"
            seen[f] = cls


def test_no_production_module_and_not_backtest_v1_imports_it():
    pat = re.compile(r"^\s*(import pit_record|from pit_record)\b", re.M)
    for p in ROOT.glob("*.py"):
        if p.name in ("pit_record.py", "conftest.py"):
            continue
        assert not pat.search(p.read_text(encoding="utf-8", errors="replace")), p.name
    for d in ("scripts",):
        for p in (ROOT / d).rglob("*.py"):
            assert not pat.search(p.read_text(encoding="utf-8", errors="replace")), str(p)


def test_a_split_after_the_date_puts_the_filings_on_the_prices_basis():
    """A split-adjusted price is in today's shares; the filing is in its own day's."""
    rows = ROWS + [("CommonStockSharesOutstanding", None, "2025-09-30", "2025-11-03", "10-Q", 100.0)]
    splits = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"],
                           "date": pd.to_datetime(["2025-06-01", "2026-03-01", "2026-03-01"]),
                           "ratio": [2.0, 10.0, 4.0]})
    assert pr.split_after(splits, "AAA", "2025-11-10") == 10.0          # only the later split
    assert pr.split_after(splits, "AAA", "2025-01-01") == 20.0
    assert pr.split_after(None, "AAA", "2025-01-01") == 1.0
    # adjusted price 2.5 = 25 / 10: the company is worth what it was worth
    rec = pr.record_as_of({"Ticker": "AAA"}, _f(rows), "2025-11-10", price=2.5, split=10.0)
    assert rec["marketCap"] == pytest.approx(2500.0)



def test_every_concept_the_record_reads_is_in_the_weekly_download():
    """A concept missing from sec_fundamentals' list is silently absent from every month."""
    import sec_fundamentals as sf
    cached = set(sf.FACT_CONCEPTS + sf.SHARE_CONCEPTS + sf.OUTSTANDING_CONCEPTS
                 + sf.DEI_SHARE_CONCEPTS + sf.EPS_CONCEPTS)
    wanted = {c for concepts in pr.C.values() for c in concepts}
    assert wanted - cached == set(), sorted(wanted - cached)


def test_a_tag_switch_inside_a_trailing_year_still_builds_it():
    """NVIDIA, 2020: the quarter tagged one way, the prior fiscal year another."""
    R1, R2 = "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"
    rows = [(R1, "2019-01-28", "2020-01-26", "2020-02-20", "10-K", 10918.0),
            (R2, "2019-01-28", "2019-04-28", "2019-05-16", "10-Q", 2220.0),
            (R2, "2020-01-27", "2020-04-26", "2020-05-21", "10-Q", 3080.0)]
    F = pr.Filed(_f(rows), "AAA", "2020-06-30")
    assert F.ttm("revenue", pd.Timestamp("2020-04-26")) == pytest.approx(10918 + 3080 - 2220)
