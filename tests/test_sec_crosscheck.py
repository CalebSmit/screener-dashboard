"""The run's inputs checked against the companies' own SEC filings (sec_crosscheck.py).

WHY THIS EXISTS - 2026-10-14. Yahoo silently stopped serving one field that week; the availability
check added the same day catches a field that vanishes, not one that changes scale or meaning.
Rebuilt from the filings, total assets agreed with Yahoo within 0.5% for 99.8% of companies, so a
field disagreeing for dozens at once is a source event. ``research/2026-10-14-yahoo-vs-sec.md``.

Pinned here: the rebuild's arithmetic (fiscal year, year-to-date combination, instant), the
first-filed rule that the FIX case forced, revenue's largest-concept rule, and that the data
loop's use of it is a **note that can never fail a run** - it reports, and calls attention past a
line written before any run was checked.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402
import sec_crosscheck as X  # noqa: E402
import write_brief as wb  # noqa: E402

T = pd.Timestamp


def fact(ticker, concept, start, end, val, filed, form="10-Q"):
    return {"ticker": ticker, "concept": concept, "start": T(start) if start else pd.NaT, "end": T(end),
            "filed": T(filed), "form": form, "fp": "", "val": float(val)}


def frame(rows):
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- the rebuild
def test_a_fiscal_year_end_takes_the_fiscal_year():
    f = frame([fact("A", "Revenues", "2025-01-01", "2025-12-31", 400, "2026-02-10", "10-K")])
    assert X.flow_ttm(f, T("2025-12-31")) == (400.0, "fiscal year")


def test_a_mid_year_quarter_combines_the_cumulative_columns():
    """TTM to June 2026 = H1 2026 + FY 2025 - H1 2025."""
    f = frame([
        fact("A", "Revenues", "2025-01-01", "2025-06-30", 180, "2025-08-01"),
        fact("A", "Revenues", "2025-01-01", "2025-12-31", 400, "2026-02-10", "10-K"),
        fact("A", "Revenues", "2026-01-01", "2026-06-30", 230, "2026-08-01"),
        fact("A", "Revenues", "2026-04-01", "2026-06-30", 120, "2026-08-01"),     # the quarter alone is not used
    ])
    v, how = X.flow_ttm(f, T("2026-06-30"))
    assert v == 230 + 400 - 180 and how == "ytd + prior fy - prior ytd"


def test_each_period_is_taken_as_first_filed():
    """FIX's Q1 2026 10-Q tags a quarter's revenue with fiscal 2025's dates. Taking the latest
    filing read it as the year; the first filing (the 10-K) has the year."""
    f = frame([
        fact("FIX", "Revenues", "2025-01-01", "2025-12-31", 8_700, "2026-02-20", "10-K"),
        fact("FIX", "Revenues", "2025-01-01", "2025-12-31", 1_831, "2026-05-01"),    # the mis-tagged comparative
    ])
    assert X.flow_ttm(f, T("2025-12-31"))[0] == 8_700


def test_missing_prior_year_columns_give_no_value_rather_than_a_guess():
    f = frame([fact("A", "Revenues", "2026-01-01", "2026-06-30", 230, "2026-08-01")])
    assert X.flow_ttm(f, T("2026-06-30")) == (None, "prior-year columns missing")


def test_a_balance_is_the_instant_value_at_the_quarter():
    f = frame([fact("A", "Assets", None, "2026-06-30", 1_000, "2026-08-01"),
               fact("A", "Assets", None, "2026-03-31", 900, "2026-05-01")])
    assert X.instant(f, T("2026-06-30")) == (1_000.0, "instant")


def test_revenue_takes_the_whole_line_not_a_component():
    """Camden's RevenueFromContractWithCustomer... is $12.6M of non-rental income."""
    f = frame([fact("CPT", "Revenues", "2025-01-01", "2025-12-31", 1_570, "2026-02-10", "10-K"),
               fact("CPT", "RevenueFromContractWithCustomerExcludingAssessedTax", "2025-01-01", "2025-12-31", 12.6,
                    "2026-02-10", "10-K")])
    v, c, _ = X.sec_value(f, X.REVENUE, "flow", T("2025-12-31"), largest=True)
    assert (v, c) == (1_570.0, "Revenues")
    # other fields take the first concept in priority order
    v2, c2, _ = X.sec_value(f, ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"), "flow",
                            T("2025-12-31"))
    assert c2 == "RevenueFromContractWithCustomerExcludingAssessedTax"


def payload(**inputs_by_ticker):
    return {"stock_detail": {t: {"inp": inp, "asof": {"is": "2025-12-31", "cf": "2025-12-31", "bs": "2025-12-31"},
                                 "sector": "X"} for t, inp in inputs_by_ticker.items()}}


def test_compare_reports_how_far_apart_and_why_a_stock_was_not_compared():
    facts = frame([fact("A", "Assets", None, "2025-12-31", 1_000, "2026-02-10", "10-K"),
                   fact("A", "PaymentsToAcquirePropertyPlantAndEquipment", "2025-01-01", "2025-12-31", 50,
                        "2026-02-10", "10-K")])
    df = X.compare(payload(A={"totalAssets": 1_100, "capex": -50}, B={"totalAssets": 5}), facts)
    a = df[(df.ticker == "A") & (df.field == "total_assets")].iloc[0]
    assert a.status == "compared" and a.rel_diff == pytest.approx(100 / 1_100)
    cap = df[(df.ticker == "A") & (df.field == "capex")].iloc[0]
    assert cap.rel_diff == 0 and cap.yahoo == 50                              # Yahoo's negative capex is a payment
    assert set(df[df.ticker == "B"].status) == {"not in SEC cache", "no yahoo value"}
    s = X.summary(df)
    assert s["total_assets"] == {"compared": 1, "within_half_pct": 0, "over_10pct": 0}


def test_the_module_is_offline_and_read_only():
    tree = ast.parse((ROOT / "sec_crosscheck.py").read_text(encoding="utf-8"))
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | \
               {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"requests", "urllib", "insider_activity", "yfinance", "factor_engine"}
    src = (ROOT / "sec_crosscheck.py").read_text(encoding="utf-8")
    assert "to_parquet" not in src and "write_text" not in src and ".to_csv(" not in src


# --------------------------------------------------------------------------- in the data loop
def _facts_and_payload(n=40, scale=1.0):
    rows, inputs = [], {}
    for i in range(n):
        t = f"T{i}"
        rows.append(fact(t, "Assets", None, "2025-12-31", 1_000 + i, "2026-02-10", "10-K"))
        inputs[t] = {"totalAssets": (1_000 + i) * scale}
    return frame(rows), payload(**inputs)


def test_the_health_note_reports_counts_and_never_fails():
    facts, pl = _facts_and_payload()
    res = h.Result()
    h.check_inputs_against_filings(pl, res, facts=facts)
    assert res.healthy and not res.failures
    note = next(n for n in res.notes if n.startswith("inputs against SEC filings: "))
    assert "total assets 40/40 within 0.5% (0 over 10%)" in note
    assert not any("ATTENTION" in n for n in res.notes)


def test_a_field_changing_scale_calls_attention_but_still_does_not_fail():
    facts, pl = _facts_and_payload(scale=1_000)                               # a field served in thousands
    res = h.Result()
    h.check_inputs_against_filings(pl, res, facts=facts)
    assert res.healthy
    assert any(n.startswith("ATTENTION - total assets differs from the companies' filings by more than 10% for 40")
               for n in res.notes)


def test_anything_going_wrong_is_a_note_not_an_exception():
    res = h.Result()
    h.check_inputs_against_filings({"stock_detail": {"A": {"inp": {"totalAssets": 1}, "asof": {"bs": "2025-12-31"}}}},
                                   res, facts=pd.DataFrame({"nonsense": [1]}))
    assert res.healthy and any(n.startswith("inputs not checked against SEC filings") for n in res.notes)


def test_the_line_and_the_fields_are_the_pre_registered_ones():
    assert h.CROSSCHECK_ATTENTION == 25
    assert h.CROSSCHECK_FIELDS == ("total_assets", "operating_cash_flow", "equity", "net_income", "revenue")
    src = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
    main = src[src.index("def main()"):]
    assert main.index("check_dispersion(payload, res)") < main.index("check_inputs_against_filings(payload, res)")
    assert "res.fail(" not in src[src.index("def check_inputs_against_filings"):src.index("def _weighted_metric_names")]


def test_the_brief_shows_an_attention_line():
    state, notable = wb.outcome(["[02:20:30] [INFO]     ok   ATTENTION - total assets differs from the filings",
                                 "[02:21:30] [INFO] === Data loop complete ==="])
    assert state == "completed" and any("ATTENTION - total assets" in n for n in notable)



# --------------------------------------------------------------------------- market value
def _share_fact(t, end, val, filed="2026-08-01"):
    return fact(t, "WeightedAverageNumberOfDilutedSharesOutstanding",
                str((pd.Timestamp(end) - pd.Timedelta(days=90)).date()), end, val, filed)


def _mv_payload(**caps):
    # The run's own date: "recent" is measured from it, so the test does not age (the 2026-10-15 lesson).
    return {"kpis": {"run_timestamp": "2026-10-14T02:00:00"},
            "stock_detail": {t: {"financials": {"market_cap": mc}, "price": px} for t, (mc, px) in caps.items()}}


def test_market_value_is_compared_with_the_latest_recent_filed_share_count():
    facts = frame([_share_fact("A", "2026-06-30", 100.0), _share_fact("A", "2026-03-31", 90.0, "2026-05-01"),
                   _share_fact("OLD", "2015-09-30", 5.0, "2015-11-01")])
    mv = X.market_value_check(_mv_payload(A=(2_100.0, 20.0), OLD=(100.0, 1.0)), facts)
    assert list(mv["ticker"]) == ["A"]                                    # a 2015 count says nothing about today
    assert mv.iloc[0]["implied"] == 105.0 and mv.iloc[0]["ratio"] == pytest.approx(1.05)


def test_a_market_value_at_the_wrong_scale_calls_attention_but_does_not_fail():
    caps = {f"T{i}": (100_000.0, 1.0) for i in range(70)}                 # served 1,000x too large
    facts = frame([_share_fact(t, "2026-06-30", 100.0) for t in caps])
    res = h.Result()
    h.check_inputs_against_filings(_mv_payload(**caps), res, facts=facts)
    assert res.healthy and not res.failures
    note = next(n for n in res.notes if n.startswith("inputs against SEC filings: "))
    assert "market value / price vs filed shares 0/70 within 5% (70 over 10%)" in note
    assert any(n.startswith("ATTENTION - market value implies a share count") for n in res.notes)
    assert h.MARKET_VALUE_ATTENTION == 60


# --------------------------------------------------------------------------- the cache's age
def _cache_at(tmp_path, monkeypatch, days_old):
    import os
    import time
    import sec_fundamentals
    facts, pl = _facts_and_payload()
    path = tmp_path / "facts.parquet"
    facts.to_parquet(path)
    stamp = time.time() - days_old * 86400
    os.utime(path, (stamp, stamp))
    monkeypatch.setattr(sec_fundamentals, "FACTS_PATH", path)
    res = h.Result()
    h.check_inputs_against_filings(pl, res)
    return res


def test_the_note_says_how_old_the_sec_cache_is(tmp_path, monkeypatch):
    res = _cache_at(tmp_path, monkeypatch, 3)
    note = next(n for n in res.notes if n.startswith("inputs against SEC filings: "))
    assert note.endswith("SEC cache 3 days old") and not any("ATTENTION" in n for n in res.notes)


def test_a_cache_that_stopped_refreshing_calls_attention_but_does_not_fail(tmp_path, monkeypatch):
    res = _cache_at(tmp_path, monkeypatch, 20)
    assert res.healthy
    assert any(n.startswith("ATTENTION - the SEC companyfacts cache is 20 days old") for n in res.notes)
    assert h.CACHE_STALE_DAYS == 14

