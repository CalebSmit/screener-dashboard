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


def test_a_10q_cannot_overwrite_a_fiscal_year_its_10k_reported():
    """Comfort Systems, 2026-10-19: its Q1 2026 10-Q tagged the quarter's revenue ($1.83bn) with a
    calendar-2025 context, after the 10-K had filed FY2025 at $9.10bn. "Latest filed" took the
    10-Q's figure and its trailing revenue came out $3.96bn against $11.23bn. A fiscal year is
    what a 10-K reports; the 10-Q's value stands only where no annual report has that year."""
    mistag = (NI, "2024-01-01", "2024-12-31", "2025-05-05", "10-Q", 110.0)   # Q1's figure, FY's context
    F = pr.Filed(_f(ROWS + [mistag]), "AAA", "2025-11-10")
    assert F.fy("net_income", pd.Timestamp("2024-12-31")) == 400.0
    assert F.ttm("net_income", pd.Timestamp("2025-09-30")) == pytest.approx(360 + 400 - 295)
    # a later 10-K/A still restates it, so restatements keep flowing
    rows = ROWS + [mistag, (NI, "2024-01-01", "2024-12-31", "2025-06-01", "10-K/A", 380.0)]
    assert pr.Filed(_f(rows), "AAA", "2025-11-10").fy("net_income", pd.Timestamp("2024-12-31")) == 380.0
    # and a year no 10-K carries is still read from the 10-Q that does
    only_q = [(NI, "2023-01-01", "2023-12-31", "2024-05-06", "10-Q", 333.0)]
    assert pr.Filed(_f(ROWS + only_q), "AAA", "2025-11-10").fy("net_income", pd.Timestamp("2023-12-31")) == 333.0


def test_quarters_and_year_to_dates_still_take_the_latest_filing():
    """The 10-K preference is for fiscal-year contexts only: a quarter restated in a later 10-Q's
    comparative column is still read at its latest version."""
    rows = ROWS + [(NI, "2025-01-01", "2025-03-31", "2026-05-04", "10-Q", 115.0)]
    F = pr.Filed(_f(rows), "AAA", "2026-06-01")
    assert F.quarter("net_income", pd.Timestamp("2025-03-31")) == pytest.approx(115.0)


def test_an_opening_balance_filed_beside_the_period_end_does_not_displace_it():
    """Air Products, 2026-10-19: its FY2018 10-K carries the 2018-09-30 balance sheet and a
    2018-10-01 opening balance (an accounting change's adoption date), both filed the same day and
    within ``TOL`` of each other. The tie was settled by an unstable sort's row order, so current
    assets came from one date and current liabilities from the other, and which moved when
    unrelated code changed. The period asked for is the period end: the nearest end wins."""
    rows = [("AssetsCurrent", None, "2018-09-30", "2018-11-20", "10-K", 5082.2),
            ("AssetsCurrent", None, "2018-10-01", "2018-11-20", "10-K", 5108.4),
            ("LiabilitiesCurrent", None, "2018-10-01", "2018-11-20", "10-K", 2364.5),
            ("LiabilitiesCurrent", None, "2018-09-30", "2018-11-20", "10-K", 2338.3)]
    end = pd.Timestamp("2018-09-30")
    for order in (rows, rows[::-1], [rows[1], rows[2], rows[0], rows[3]]):
        F = pr.Filed(_f(order), "AAA", "2018-11-30")
        assert F._instant1("AssetsCurrent", end) == 5082.2
        assert F._instant1("LiabilitiesCurrent", end) == 2338.3
    # a later filing still wins over a nearer end: restatements flow as before
    later = rows + [("AssetsCurrent", None, "2018-10-01", "2019-01-25", "10-Q", 5110.0)]
    assert pr.Filed(_f(later), "AAA", "2019-02-28")._instant1("AssetsCurrent", end) == 5110.0


def test_the_record_does_not_depend_on_the_order_of_the_facts():
    """Every tie is broken by a value, never by row order (stable sorts, an amendment after its
    original on the same day): shuffling the facts cannot change a record."""
    rows = ROWS + [("Assets", None, "2025-10-01", "2025-11-03", "10-Q", 5100.0),
                   (NI, "2024-01-01", "2024-12-31", "2025-02-10", "10-K/A", 401.0)]
    base = {"Ticker": "AAA"}
    ref = pr.record_as_of(base, _f(rows), "2025-11-10", price=10.0)
    for seed in range(5):
        shuffled = _f(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        rec = pr.record_as_of(base, shuffled, "2025-11-10", price=10.0)
        for k, v in ref.items():
            if isinstance(v, float) and np.isnan(v):
                assert isinstance(rec[k], float) and np.isnan(rec[k]), k
            else:
                assert rec[k] == v, (seed, k, v, rec[k])
    assert pr.Filed(_f(rows), "AAA", "2025-11-10").fy("net_income", pd.Timestamp("2024-12-31")) == 401.0


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
    # No returnOnEquity / returnOnAssets since 2026-10-20: compute_metrics stopped reading the
    # source's ratios, so market_fields no longer builds a third construction of them on ENDING
    # balances. The bank ROE and ROA are statement fields now (tests/test_bank_roe_roa.py), which
    # statement_fields restates as of the month being scored.
    assert "returnOnEquity" not in rec and "returnOnAssets" not in rec


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


# v2 is the consumer the module exists for; nothing else at the root may read it.
V2_MODULES = ("backtest_v2.py", "backtest_v2_returns.py")


def test_no_production_module_and_not_backtest_v1_imports_it():
    pat = re.compile(r"^\s*(import pit_record|from pit_record)\b", re.M)
    assert not pat.search((ROOT / "backtest.py").read_text(encoding="utf-8"))   # v1, by name
    for p in ROOT.glob("*.py"):
        if p.name in ("pit_record.py", "conftest.py") + V2_MODULES:
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


def test_the_methodology_page_quotes_the_measured_lookahead():
    """The page's point-in-time limitation quotes the 2026-10-15 measurement, from its JSON, and
    says the rebuild sets no weight (claim ``overview.lookahead_size``)."""
    import json
    import yaml
    import run_screener
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    text = run_screener.build_screener_overview(cfg)
    m = json.loads((ROOT / "research" / "measurements" / "2026-10-15-lookahead-fundamentals-half.json")
                   .read_text(encoding="utf-8"))
    pm = m["per_month"]                                 # exact, not a rounded fraction rounded again
    share = (sum(r["fundamentals_half"]["decile_changed"] for r in pm)
             / sum(r["fundamentals_half"]["n"] for r in pm))
    assert f"**{share * 100:.1f}% of stock-months in a different tenth** of the ranking" in text
    assert "It does not set any weight on this page" in text
    assert "It does not reconstruct what was known at a past date" not in text


def test_operating_income_is_rebuilt_before_restructuring_and_impairment_charges():
    """The live source's "Operating Income" leaves out special charges (2026-10-15,
    research/2026-10-15-earnings-definition.md), so v2 adds the filed charges back to the filed
    operating income - a combined restructuring-and-impairment tag standing for its parts."""
    rows = ROWS + [
        ("OperatingIncomeLoss", "2024-01-01", "2024-12-31", "2025-02-10", "10-K", 300.0),
        ("GoodwillImpairmentLoss", "2024-01-01", "2024-12-31", "2025-02-10", "10-K", 50.0),
        ("RestructuringCharges", "2024-01-01", "2024-12-31", "2025-02-10", "10-K", 20.0),
    ]
    s = pr.statement_fields(_f(rows), "AAA", "2025-03-01")
    assert s["ebit"] == pytest.approx(370.0)               # 300 + 50 + 20
    both = rows + [("RestructuringCostsAndAssetImpairmentCharges", "2024-01-01", "2024-12-31",
                    "2025-02-10", "10-K", 60.0)]
    assert pr.statement_fields(_f(both), "AAA", "2025-03-01")["ebit"] == pytest.approx(360.0)  # never both
    plain = ROWS + [("OperatingIncomeLoss", "2024-01-01", "2024-12-31", "2025-02-10", "10-K", 300.0)]
    assert pr.statement_fields(_f(plain), "AAA", "2025-03-01")["ebit"] == pytest.approx(300.0)


def test_d_and_a_falls_back_to_its_two_parts():
    rows = ROWS + [("Depreciation", "2024-01-01", "2024-12-31", "2025-02-10", "10-K", 30.0),
                   ("AmortizationOfIntangibleAssets", "2024-01-01", "2024-12-31", "2025-02-10", "10-K", 5.0)]
    assert pr.statement_fields(_f(rows), "AAA", "2025-03-01")["da_cf"] == pytest.approx(35.0)
