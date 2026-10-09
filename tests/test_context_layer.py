"""The context layer: trend, options, insiders, rates, the market backdrop and the track record.

WHY THIS EXISTS - 2026-10-08 (owner; ``plan/context-layer.md``). The owner asked for technicals,
macro and options data beside the score, plus a track record and insider buying, with one rule
settled up front: **context only - shown beside the score, never in it, and recorded every run so
each signal builds its own out-of-sample record.**

These tests hold each calculation to an answer known by construction (no network), and hold the
layer to that rule: nothing here reaches a scored field, no copy reads as advice, and every number
the page states about how something is computed is true of the code.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import context_signals as cs  # noqa: E402
import insider_activity as ia  # noqa: E402
import market_context as mc  # noqa: E402
import track_record as tr  # noqa: E402

PAYLOAD = ROOT / "dashboard_data.js"


# ---------------------------------------------------------------------------
# price context
# ---------------------------------------------------------------------------

def _closes(n=300, start=100.0, step=0.5, end="2026-10-07"):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.Series(start + step * np.arange(n), index=idx)


def test_moving_averages_and_range_are_exact():
    c = _closes()
    out = cs.price_context(c, pd.Series(1000.0, index=c.index))
    assert out["_ctx_last_close"] == pytest.approx(c.iloc[-1])
    assert out["_ctx_sma50"] == pytest.approx(c.tail(50).mean())
    assert out["_ctx_sma200"] == pytest.approx(c.tail(200).mean())
    assert out["_ctx_sma200_prev"] == pytest.approx(c.iloc[-220:-20].mean())
    assert out["_ctx_high_52w"] == pytest.approx(c.tail(252).max())
    assert out["_ctx_low_52w"] == pytest.approx(c.tail(252).min())
    assert out["_ctx_vol_ratio"] == pytest.approx(1.0)


def test_recent_returns_use_calendar_lookbacks():
    c = _closes()
    out = cs.price_context(c)
    target = c.index[-1] - pd.Timedelta(days=30)
    prior = c.loc[c.index <= target].iloc[-1]
    assert out["_ctx_ret_1m"] == pytest.approx(c.iloc[-1] / prior - 1)


def test_weekly_series_samples_the_daily_averages():
    c = _closes()
    wk = json.loads(cs.price_context(c)["_ctx_weekly"])
    assert len(wk["d"]) == len(wk["c"]) == len(wk["s50"]) == len(wk["s200"]) <= cs.WEEKS
    # last weekly point is the last close and the last 50-day average
    # stored to 4 significant digits (payload size), so within half a unit in the 4th digit
    assert wk["c"][-1] == pytest.approx(c.iloc[-1], rel=5e-4)
    assert wk["s50"][-1] == pytest.approx(c.tail(50).mean(), rel=5e-4)


def test_short_history_omits_rather_than_estimates():
    out = cs.price_context(_closes(n=60))
    assert "_ctx_sma50" in out and "_ctx_sma200" not in out
    assert cs.price_context(_closes(n=10)) == {}


def test_trend_state_reads_both_averages():
    assert cs.trend_state({"_ctx_last_close": 110, "_ctx_sma50": 105, "_ctx_sma200": 100}) == "uptrend"
    assert cs.trend_state({"_ctx_last_close": 90, "_ctx_sma50": 95, "_ctx_sma200": 100}) == "downtrend"
    assert cs.trend_state({"_ctx_last_close": 105, "_ctx_sma50": 95, "_ctx_sma200": 100}) == "mixed"


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

def test_expiry_spans_a_nearby_report():
    today = date(2026, 10, 8)
    exps = ["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-20", "2026-12-18"]
    exp, spans = cs.choose_expiry(exps, today, date(2026, 11, 4))
    assert exp == "2026-11-06" and spans is True


def test_expiry_without_a_report_is_nearest_thirty_days():
    today = date(2026, 10, 8)
    exps = ["2026-10-09", "2026-10-23", "2026-11-06", "2026-11-20"]
    exp, spans = cs.choose_expiry(exps, today, None)
    assert exp == "2026-11-06" and spans is False      # 29 days out, nearest to 30
    assert cs.choose_expiry(["2026-10-09"], today, None) == (None, False)   # 1 day: too short


def _chain(spot=100.0, call_bid=3.0, call_ask=3.2, put_bid=2.8, put_ask=3.0, iv=0.30, skew_iv=0.38):
    strikes = [85, 90, 95, 100, 105, 110]
    calls = pd.DataFrame({"strike": strikes, "bid": [16, 11, 6.5, call_bid, 1.2, 0.5],
                          "ask": [16.4, 11.4, 6.9, call_ask, 1.4, 0.7],
                          "impliedVolatility": [0.33, 0.32, 0.31, iv, 0.29, 0.28],
                          "openInterest": [10, 50, 200, 500, 300, 100]})
    puts = pd.DataFrame({"strike": strikes, "bid": [0.4, 0.8, 1.5, put_bid, 6.0, 10.5],
                         "ask": [0.6, 1.0, 1.7, put_ask, 6.4, 10.9],
                         "impliedVolatility": [0.42, skew_iv, 0.34, iv, 0.29, 0.28],
                         "openInterest": [100, 400, 300, 600, 50, 10]})
    return calls, puts


def test_expected_move_is_the_atm_straddle_over_the_price():
    calls, puts = _chain()
    out = cs.options_context(calls, puts, 100.0, "2026-11-06", date(2026, 10, 8), True)
    assert out["_ctx_opt_status"] == "ok"
    assert out["_ctx_opt_move"] == pytest.approx((3.1 + 2.9) / 100.0)
    assert out["_ctx_opt_iv"] == pytest.approx(0.30)
    assert out["_ctx_opt_skew"] == pytest.approx(0.38 - 0.30)
    assert out["_ctx_opt_pc_oi"] == pytest.approx(1460 / 1160)
    assert out["_ctx_opt_days"] == 29 and out["_ctx_opt_spans_earnings"] is True


def test_stale_or_wide_quotes_give_no_move():
    calls, puts = _chain(call_bid=0.0, call_ask=0.0)
    out = cs.options_context(calls, puts, 100.0, "2026-11-06", date(2026, 10, 8), False)
    assert "_ctx_opt_move" not in out and out["_ctx_opt_status"] in ("partial", "stale-quotes")
    calls, puts = _chain(call_bid=1.0, call_ask=6.0)        # spread far wider than the mid
    out = cs.options_context(calls, puts, 100.0, "2026-11-06", date(2026, 10, 8), False)
    assert "_ctx_opt_move" not in out


def test_implausible_iv_is_dropped():
    calls, puts = _chain(iv=9.0)
    out = cs.options_context(calls, puts, 100.0, "2026-11-06", date(2026, 10, 8), False)
    assert "_ctx_opt_iv" not in out


def test_only_the_next_report_field_picks_an_expiry():
    # `earningsTimestamp` is the LAST report for some tickers - it must never be read.
    assert cs.earnings_date_from_info({"earningsTimestamp": 1_800_000_000}) is None
    assert cs.earnings_date_from_info({"earningsTimestampStart": 1_800_000_000}) is not None


# ---------------------------------------------------------------------------
# rate sensitivity
# ---------------------------------------------------------------------------

def test_rate_beta_recovers_a_known_slope():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2025-09-01", periods=260)
    dy = pd.Series(rng.normal(0, 0.06, len(idx)), index=idx)
    r = -0.05 * dy + rng.normal(0, 0.001, len(idx))
    out = cs.rate_sensitivity({d.strftime("%Y-%m-%d"): v for d, v in r.items()}, dy)
    assert out["beta"] == pytest.approx(-0.05, abs=0.005)
    assert 0.5 < out["r2"] <= 1 and out["n"] == 260


def test_rate_beta_needs_enough_common_days():
    idx = pd.bdate_range("2026-01-01", periods=50)
    assert cs.rate_sensitivity({d.strftime("%Y-%m-%d"): 0.01 for d in idx}, pd.Series(0.01, index=idx)) == {}


# ---------------------------------------------------------------------------
# market backdrop
# ---------------------------------------------------------------------------

def test_read_series_and_summarise():
    idx = pd.date_range("2016-01-01", "2026-10-01", freq="B")
    s = pd.Series(np.linspace(1, 5, len(idx)), index=idx)
    text = "observation_date,DGS10\n" + "\n".join(f"{d.date()},{v:.4f}" for d, v in s.items())
    back = mc.read_series_csv(text)
    out = mc.summarise("DGS10", back)
    assert out["last"] == pytest.approx(5.0, abs=1e-3)
    assert out["pct_10y"] > 99           # the latest of a rising series is at the top of its range
    assert out["chg_1y"] > 0 and len(out["spark"]) == len(out["spark_d"])


def test_cpi_is_shown_year_over_year():
    idx = pd.date_range("2015-01-01", periods=140, freq="MS")
    cpi = pd.Series(100 * 1.03 ** (np.arange(140) / 12), index=idx)
    out = mc.summarise("CPIAUCSL", cpi)
    assert out["last"] == pytest.approx(3.0, abs=0.01)


def test_sahm_indicator_definition():
    u = pd.Series([4.0] * 13 + [4.0, 4.6, 5.2], index=pd.date_range("2024-01-01", periods=16, freq="MS"))
    expected = np.mean([4.0, 4.6, 5.2]) - 4.0
    assert mc.sahm_indicator(u) == pytest.approx(round(expected, 2))


def test_readings_name_their_rule():
    summ = {"T10Y3M": {"last": -0.4}, "VIXCLS": {"last": 30.0, "median_10y": 17.0},
            "BAA10Y": {"last": 2.5, "pct_10y": 80}, "CPIAUCSL": {"last": 3.1, "date": "2026-08-01"},
            "UNRATE": {"last": 4.4}}
    rs = {r["k"]: r for r in mc.readings(summ, 0.6)}
    assert rs["curve"]["state"] == "inverted" and "Estrella" in rs["curve"]["text"]
    assert rs["vol"]["state"] == "elevated"
    assert rs["credit"]["state"] == "wide"
    assert rs["jobs"]["state"] == "triggered" and "Sahm" in rs["jobs"]["text"]


# ---------------------------------------------------------------------------
# insiders
# ---------------------------------------------------------------------------

def _yahoo_rows(today):
    d = lambda k: pd.Timestamp(today - timedelta(days=k))  # noqa: E731
    return pd.DataFrame([
        {"Shares": 1000, "Value": 50000.0, "Text": "Purchase at price 50.00 per share.", "Insider": "ALPHA ANN", "Position": "Chief Executive Officer", "Start Date": d(5)},
        {"Shares": 500, "Value": 25000.0, "Text": "Purchase at price 50.00 per share.", "Insider": "BETA BOB", "Position": "Director", "Start Date": d(20)},
        {"Shares": 400, "Value": 20000.0, "Text": "Purchase at price 50.00 per share.", "Insider": "GAMMA GIL", "Position": "Director", "Start Date": d(40)},
        {"Shares": 900, "Value": 90000.0, "Text": "Sale at price 100.00 per share.", "Insider": "DELTA DEE", "Position": "Officer", "Start Date": d(10)},
        {"Shares": 3000, "Value": 150000.0, "Text": "Stock Award(Grant) at price 50.00 per share.", "Insider": "ALPHA ANN", "Position": "Chief Executive Officer", "Start Date": d(3)},
        {"Shares": 200, "Value": float("nan"), "Text": "", "Insider": "EPS EVE", "Position": "Officer", "Start Date": d(2)},
        {"Shares": 999, "Value": 99900.0, "Text": "Purchase at price 100.00 per share.", "Insider": "OLD OLLY", "Position": "Director", "Start Date": d(150)},
    ])


def test_grants_and_exercises_are_not_purchases():
    today = date(2026, 10, 8)
    rows = ia.rows_from_yahoo(_yahoo_rows(today), today)
    codes = sorted(r["code"] for r in rows)
    assert codes == ["P", "P", "P", "P", "S"]          # the award and the blank row are gone


def test_ninety_day_summary_and_cluster():
    today = date(2026, 10, 8)
    s = ia.summarise_rows(ia.rows_from_yahoo(_yahoo_rows(today), today), today)
    assert s["buy_n"] == 3 and s["buy_people"] == 3 and s["buy_value"] == 95000.0   # OLLY is 150 days old
    assert s["cluster"] is True and s["officer_buy"] is True
    assert s["sell_n"] == 1 and s["sell_value"] == 90000.0
    assert s["recent"][0]["date"] >= s["recent"][-1]["date"]


def test_form4_xml_parsing():
    xml = """<?xml version="1.0"?><ownershipDocument><aff10b5One>1</aff10b5One>
      <reportingOwner><reportingOwnerId><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
      <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>CFO</officerTitle></reportingOwnerRelationship></reportingOwner>
      <nonDerivativeTable><nonDerivativeTransaction><transactionDate><value>2026-09-30</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>100</value></transactionShares><transactionPricePerShare><value>12.5</value></transactionPricePerShare>
      <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"""
    f = ia.parse_form4(xml)
    assert f["plan"] is True and f["owners"][0] == {"name": "Doe Jane", "role": "CFO"}
    assert f["trades"][0] == {"date": "2026-09-30", "code": "S", "shares": 100.0, "price": 12.5, "ad": "D", "value": 1250.0}


# ---------------------------------------------------------------------------
# track record
# ---------------------------------------------------------------------------

class _Snap:
    def __init__(self, d, ranks):
        self.date, self.ranks = d, ranks


def _prices():
    idx = pd.bdate_range("2026-01-02", "2026-03-31")
    n = len(idx)
    data = {f"T{i}": 100 * (1 + 0.001 * i) ** np.arange(n) for i in range(10)}
    data["RSP"] = 100 * 1.0005 ** np.arange(n)
    data["SPY"] = 100 * 1.0007 ** np.arange(n)
    return pd.DataFrame(data, index=idx)


def test_rebalances_on_the_first_run_of_each_month():
    assert tr.rebalance_dates(["2026-01-05", "2026-01-20", "2026-02-02", "2026-02-15", "2026-04-01"]) == \
        ["2026-01-05", "2026-02-02", "2026-04-01"]


def test_basket_return_matches_hand_arithmetic(monkeypatch):
    monkeypatch.setattr(tr, "TOP_N", 2)
    prices = _prices()
    ranks = {f"T{i}": 10 - i for i in range(10)}      # T9 ranks 1st, T8 2nd
    out = tr.build([_Snap("2026-01-05", ranks)], prices, date(2026, 3, 31), changelog=[])
    entry = prices.loc[prices.index >= "2026-01-05"].iloc[0]
    last = prices.iloc[-1]
    expected = 0.5 * (last["T9"] / entry["T9"]) + 0.5 * (last["T8"] / entry["T8"]) - 1
    assert out["total"]["top"] == pytest.approx(expected, abs=1e-5)
    assert out["total"]["RSP"] == pytest.approx(last["RSP"] / entry["RSP"] - 1, abs=1e-5)
    assert out["spread"] > 0                               # the top fifth rose faster by construction


def test_entry_is_never_before_the_run():
    prices = _prices()
    out = tr.build([_Snap("2026-01-10", {f"T{i}": i + 1 for i in range(10)})], prices, date(2026, 3, 31), changelog=[])
    assert out["periods"][0]["entry"] >= "2026-01-10"      # a Saturday run enters on Monday's close


def test_a_name_whose_prices_stop_is_held_as_cash():
    prices = _prices()
    prices.loc[prices.index > "2026-02-01", "T9"] = np.nan
    v, info = tr.run_basket(prices, ["T9", "T8"], pd.Timestamp("2026-01-05"), None, 100.0)
    assert "T9" in info["stale"] and v.notna().all()


def test_no_current_holdings_list_is_published():
    """The Model Portfolio panel was removed (2026-08-26) because a buy list reads as advice."""
    prices = _prices()
    out = tr.build([_Snap("2026-01-05", {f"T{i}": i + 1 for i in range(10)})], prices, date(2026, 3, 31), changelog=[])
    assert "holdings" not in json.dumps(out)
    for p in out["periods"]:       # periods carry returns, never the list of names held
        assert all(not isinstance(v, list) or k == "stale" for k, v in p.items()), p


# ---------------------------------------------------------------------------
# the record of every signal
# ---------------------------------------------------------------------------

def test_context_log_writes_one_file_per_date(tmp_path, monkeypatch):
    raw = pd.DataFrame({"Ticker": ["A", "B"], "_ctx_ret_1m": [0.1, -0.2], "_ctx_weekly": ["{}", "{}"],
                        "_ctx_insider": [json.dumps([]), json.dumps([])], "marketCap": [1, 2]})
    run = tmp_path / "run"
    run.mkdir()
    raw.to_parquet(run / "00_raw_fetch.parquet")
    monkeypatch.setattr(cs, "CONTEXT_LOG_DIR", tmp_path / "log")
    assert cs.write_context_log(run, "2026-10-08") == 2
    assert cs.write_context_log(run, "2026-10-08") == 2
    files = list((tmp_path / "log").glob("*.parquet"))
    assert [f.name for f in files] == ["2026-10-08.parquet"]
    df = pd.read_parquet(files[0])
    assert "marketCap" not in df.columns and "_ctx_weekly" not in df.columns and "_ctx_ins_buy_n" in df.columns


# ---------------------------------------------------------------------------
# the rule: context never reaches a score, and never reads as advice
# ---------------------------------------------------------------------------

def test_no_scoring_module_reads_a_context_field():
    for name in ("factor_engine.py", "calc_trace.py", "portfolio_constructor.py", "improvement_engine.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        # factor_engine *writes* _ctx_ fields at fetch; nothing may read one back into a metric.
        reads = re.findall(r"""(?:row|df|rec|r)\[["'](_ctx_[a-z0-9_]+)["']\](?!\s*=)""", src)
        assert not reads, f"{name} reads context fields {reads}"


@pytest.fixture(scope="module")
def payload():
    if not PAYLOAD.exists():
        pytest.skip("dashboard_data.js not present")
    t = PAYLOAD.read_text(encoding="utf-8", errors="replace")
    d = json.loads(t[t.find("{"):t.rfind("}") + 1])
    if not any(s.get("ctx") for s in d["stock_detail"].values()):
        pytest.skip("payload predates the context layer")
    return d


def test_context_is_display_only_in_the_payload(payload):
    for s in payload["stock_detail"].values():
        assert "ctx" not in s["raw"] and "ctx" not in s["pct"]
        assert not any(k.startswith("_ctx") for k in s["raw"]) and not any(k.startswith("_ctx") for k in s["pct"])


def test_context_covers_most_of_the_universe(payload):
    sd = payload["stock_detail"]
    have = sum(1 for s in sd.values() if (s.get("ctx") or {}).get("s200"))
    assert have / len(sd) > 0.9, f"trend context for only {have} of {len(sd)}"


def test_new_copy_carries_no_advice_language():
    import generate_dashboard as gd
    import stock_summary
    texts = [gd._js_context()]
    texts += [n["text"] for n in mc.FACTOR_NOTES]
    texts += [r["text"] for r in mc.readings({"T10Y3M": {"last": 1.0}, "VIXCLS": {"last": 15, "median_10y": 17},
                                              "BAA10Y": {"last": 1.5, "pct_10y": 10}, "CPIAUCSL": {"last": 3.0, "date": "2026-08-01"},
                                              "UNRATE": {"last": 4.2}}, 0.1)]
    strings = []
    for t in texts:
        strings += re.findall(r"'([^'\n]{12,})'", t) if "function" in t else [t]
    hits = [x for x in strings if stock_summary.advice_terms_in(x)]
    # the one permitted pattern is the sentence that *refuses* advice ("none of it says when to buy or sell")
    hits = [x for x in hits if "none of it" not in x.lower() and "not a portfolio" not in x.lower()
            and "none of them" not in x.lower()]
    assert not hits, hits


def test_the_data_loop_commits_the_context_outputs():
    ps = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8")
    block = ps[ps.index("$DataArtifacts"):]
    block = block[:block.index(")")]
    for a in ("data/market_context.json", "data/track_record.json", "data/context_log"):
        assert a in block, a
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for c in ("data/market/", "data/track/", "data/insider/"):
        assert c in gi, c


# ---------------------------------------------------------------------------
# the context pass never costs the core fetch
# ---------------------------------------------------------------------------

def test_options_and_insider_calls_are_not_in_the_core_fetch():
    """Inside the core fetch they tripped the rate limiter (2026-10-08) and slowed scored data."""
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    body = src[src.index("def _fetch_single_ticker_inner"):]
    body = body[:body.index("\ndef ", 10)]
    assert "option_chain" not in body and "insider_transactions" not in body


def test_context_pass_skips_failed_core_fetches_and_respects_the_budget(monkeypatch):
    import context_fetch as cf
    calls = []
    monkeypatch.setattr(cf, "_context_for", lambda t, p, e, d: calls.append(t) or {"_ctx_opt_status": "ok"})
    monkeypatch.setattr(cf, "PACE_SECONDS", 0)
    raw = [{"Ticker": "A", "price_latest": 10}, {"Ticker": "B", "_error": "x"}, {"Ticker": "C", "price_latest": 5}]
    st = cf.enrich(raw, budget_seconds=60, log=lambda *a: None)
    assert sorted(calls) == ["A", "C"] and st["done"] == 2 and "_ctx_opt_status" not in raw[1]
    st = cf.enrich([{"Ticker": "Z"}], budget_seconds=-1, log=lambda *a: None)
    assert st["stopped"] == "time budget" and st["done"] == 0


def test_context_pass_stops_when_rate_limited(monkeypatch):
    import context_fetch as cf

    def limited(*a):
        raise RuntimeError("Too Many Requests. Rate limited.")
    monkeypatch.setattr(cf, "_context_for", limited)
    monkeypatch.setattr(cf, "PACE_SECONDS", 0)
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    raw = [{"Ticker": f"T{i}", "price_latest": 1} for i in range(40)]
    st = cf.enrich(raw, budget_seconds=60, log=lambda *a: None)
    assert st["stopped"] == "rate limited" and st["done"] == 0
