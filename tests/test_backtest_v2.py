"""Backtest v2's rankings: every input is what was known at the month-end (``backtest_v2.py``).

Pinned: no window reads past the month-end, a member that cannot be scored is counted with its
reason and never silently dropped, the inputs with no free history are removed from the weights
openly rather than frozen at today's values, the momentum and risk inputs are computed by the live
fetch's own function (and reproduce it exactly where the cache allows the check), and nothing in
production imports the module.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest_v2 as v2  # noqa: E402
import factor_engine as fe  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the weights
def test_the_unmeasurable_inputs_carry_no_weight_and_say_so():
    c = v2.v2_config(CFG)
    for table in ("metric_weights", "bank_metric_weights"):
        for metrics in (c.get(table) or {}).values():
            for m, w in metrics.items():
                if m in v2.UNMEASURABLE_METRICS:
                    assert w == 0, m
    assert c["factor_weights"]["revisions"] == 0          # nothing left in it
    for cat in ("valuation", "quality", "growth", "momentum", "risk", "size", "investment"):
        assert c["factor_weights"][cat] == CFG["factor_weights"][cat]
    assert CFG["factor_weights"]["revisions"] > 0          # the live config is untouched


def test_every_unmeasurable_metric_is_a_real_metric():
    assert set(v2.UNMEASURABLE_METRICS) <= set(fe.METRIC_COLS)


# --------------------------------------------------------------------------- one computation
def test_the_fetch_and_v2_share_one_price_history_function():
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    body = src[src.index("def _fetch_single_ticker_inner"):]
    body = body[:body.index("\ndef ", 10)]
    assert "price_history_fields(hist)" in body
    assert "daily_ret.std()" not in body                   # not a second copy inline
    assert "fe.price_history_fields(" in (ROOT / "backtest_v2.py").read_text(encoding="utf-8")


def _hist(n=300, seed=1, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Close": closes, "Volume": np.full(n, 1e6)}, index=idx)


def test_price_history_fields_is_the_fetch_arithmetic():
    h = _hist()
    f = fe.price_history_fields(h)
    lr = np.log(h["Close"] / h["Close"].shift(1)).dropna()
    assert f["price_latest"] == pytest.approx(h["Close"].iloc[-1])
    assert f["volatility_1y"] == pytest.approx(lr.std() * np.sqrt(252))
    assert f["_vol_daily_sd"] == pytest.approx(lr.std())
    assert len(f["_daily_returns"]) == len(lr)
    target = h.index[-1] - pd.Timedelta(days=365)
    assert f["price_12m_ago"] == pytest.approx(h["Close"][h.index <= target].iloc[-1])
    assert f["avg_daily_dollar_volume"] == pytest.approx((h["Close"] * h["Volume"]).tail(63).mean())


def test_a_mixed_scale_series_is_withheld_not_scored():
    h = _hist()
    h.iloc[150:, 0] = h.iloc[150:, 0] / 10                 # a 10:1 split the history did not adjust
    splits = pd.DataFrame({"ticker": ["AAA"], "date": [h.index[150]], "ratio": [10.0]})
    g = pd.DataFrame({"adj_close": h["Close"], "volume": h["Volume"]})
    f = fe.price_history_fields(v2.history_window(g, h.index[-1], splits))
    assert "_price_series_rejected" in f
    for k in fe.PRICE_SERIES_DERIVED_FIELDS:
        assert np.isnan(f[k]) if not isinstance(f[k], dict) else False


@pytest.fixture(autouse=True)
def _no_writes_to_the_real_metrics_cache(monkeypatch, tmp_path):
    """build_month caches each month's metrics; a test's synthetic month must never land in
    cache/v2_metrics/, where the research engine would read it (found 2026-10-15: 2024-06-28)."""
    monkeypatch.setattr(v2, "METRICS_DIR", tmp_path / "v2_metrics")


# --------------------------------------------------------------------------- no look-ahead
def _inputs(tickers=("AAA",), with_base=True, ciks=None, month_ciks=None):
    days = pd.bdate_range("2023-01-02", "2025-12-31")
    rows = []
    for t in tickers:
        px = np.linspace(50, 80, len(days))
        rows += [{"ticker": t, "date": d, "close": p, "adj_close": p * 0.98, "volume": 1e6}
                 for d, p in zip(days, px)]
    daily = pd.DataFrame(rows)
    idx_days = pd.bdate_range("2019-01-02", "2025-12-31")
    index = pd.concat([pd.DataFrame({"ticker": s, "date": idx_days,
                                     "close": np.linspace(1000, 2000, len(idx_days)) if s != "^IRX"
                                     else np.where(idx_days <= "2024-06-30", 5.0, 3.0)})
                       for s in ("^SP500TR", "^GSPC", "^IRX")])
    facts = pd.DataFrame([{"ticker": "AAA", "concept": "NetIncomeLoss", "start": pd.Timestamp("2023-01-01"),
                           "end": pd.Timestamp("2023-12-31"), "filed": pd.Timestamp("2024-02-10"),
                           "form": "10-K", "fp": "FY", "val": 100.0}])
    base = {t: {"Ticker": t, "sector": "Industrials"} for t in tickers} if with_base else {}
    return v2.Inputs(daily, index, pd.DataFrame(columns=["ticker", "date", "ratio"]), facts, base, ciks, month_ciks)


def test_the_history_window_ends_at_the_month_end():
    inp = _inputs()
    w = v2.history_window(inp.daily["AAA"], pd.Timestamp("2024-06-28"))
    assert w.index.max() <= pd.Timestamp("2024-06-28")
    assert w.index.min() > pd.Timestamp("2024-06-28") - pd.DateOffset(months=13)


def test_market_inputs_read_nothing_after_the_month_end():
    inp = _inputs()
    mr, rf, state = v2.market_inputs(inp, pd.Timestamp("2024-06-28"), {"enabled": True})
    assert mr.index.max() <= pd.Timestamp("2024-06-28")
    assert rf == pytest.approx(0.05)                        # the later 3.0 is invisible
    assert state["state"] in ("NORMAL", "PANIC", "UNKNOWN")


def test_a_member_that_cannot_be_scored_says_why():
    inp = _inputs(("AAA",))
    assert v2.record_at("ZZZ", inp, pd.Timestamp("2024-06-28")) == (None, "no_current_record")
    assert v2.record_at("AAA", inp, pd.Timestamp("2022-06-30")) == (None, "no_price")
    rec, why = v2.record_at("AAA", inp, pd.Timestamp("2024-06-28"))
    assert why is None
    assert rec["currentPrice"] == pytest.approx(inp.daily["AAA"].loc[:"2024-06-28", "close"].iloc[-1])
    assert rec["netIncome"] == 100.0                        # filed 2024-02-10, visible
    for k in ("forwardEps", "targetMeanPrice", "shortRatio", "_fy1_eps_current"):
        assert np.isnan(rec[k])                             # never today's value


def test_the_census_accounts_for_every_member(monkeypatch):
    inp = _inputs(("AAA",))

    class Snap:
        tickers = ("AAA", "GONE")
        n_members = 2
        staleness_days = 0
    monkeypatch.setattr(v2.uh, "members_on", lambda as_of, allow_network=False: Snap)
    monkeypatch.setattr(v2, "score_month", lambda recs, cfg, mr, rf, st: pd.DataFrame(
        {"Ticker": [r["Ticker"] for r in recs], "Composite": [50.0] * len(recs), "Rank": [1] * len(recs)}))
    out, census = v2.build_month("2024-06-28", inp, v2.v2_config(CFG))
    assert census["members"] == 2 and census["built"] == 1
    assert census["not_scoreable"] == {"no_current_record": ["GONE"]}
    assert census["built"] + sum(len(v) for v in census["not_scoreable"].values()) == census["members"]


def test_a_renamed_member_is_found_under_its_current_symbol(monkeypatch):
    """2026-10-15: v2's inputs are keyed by today's symbols, so a member listed under an old one
    (FB in 2020, today META) was counted missing - 635 member-months of companies that never left
    the index. The SEC registrant number survives the rename. Two old symbols never claim one
    company, and a symbol that is itself a member that month is never borrowed."""
    inp = _inputs(("AAA",), ciks={"OLD": 7, "OLD2": 7, "AAA": 7, "GONE": 9})

    def snap(*tickers):
        class Snap:
            n_members = len(tickers)
            staleness_days = 0
        Snap.tickers = tickers
        monkeypatch.setattr(v2.uh, "members_on", lambda as_of, allow_network=False: Snap)

    snap("OLD", "GONE")
    _, census = v2.build_month("2024-06-28", inp, v2.v2_config(CFG))
    assert census["renamed"] == {"OLD": "AAA"} and census["built"] == 1
    assert census["not_scoreable"] == {"no_current_record": ["GONE"]}
    snap("OLD", "OLD2")
    _, census = v2.build_month("2024-06-28", inp, v2.v2_config(CFG))
    assert census["renamed"] == {"OLD": "AAA"} and census["not_scoreable"] == {"no_current_record": ["OLD2"]}
    snap("OLD", "AAA")
    _, census = v2.build_month("2024-06-28", inp, v2.v2_config(CFG))
    assert census["renamed"] == {} and census["not_scoreable"] == {"no_current_record": ["OLD"]}


# --------------------------------------------------------------------------- exactness on real data
def test_the_cached_history_reproduces_the_live_fetch_exactly():
    """Where the cache and a run's raw fetch overlap, the shared function on the cached window
    must give the fetch's own numbers (measured 2026-10-15: 100% of ~500 within 0.5%)."""
    daily_path = ROOT / "cache" / "v2_daily.parquet"
    runs = sorted((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime)
    if not daily_path.exists() or not runs:
        pytest.skip("no cached daily history or run artifact on this machine")
    raw = pd.read_parquet(runs[-1]).set_index("Ticker")
    daily = pd.read_parquet(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])
    last = pd.Timestamp(str(raw["_ctx_last_date"].dropna().max()))
    if daily["date"].max() < last:
        pytest.skip("the cache ends before the newest run")
    checked = 0
    for t in ("AAPL", "MSFT", "JPM", "XOM", "NVDA", "KO", "AMZN", "UNH"):
        if t not in raw.index:
            continue
        g = daily[daily["ticker"] == t].set_index("date").sort_index()
        f = fe.price_history_fields(v2.history_window(g, last))
        for k in ("price_latest", "price_12m_ago", "volatility_1y"):
            assert f[k] == pytest.approx(float(raw.loc[t, k]), rel=5e-3), (t, k)
        checked += 1
    assert checked >= 5


def test_no_production_module_imports_it():
    pat = re.compile(r"^\s*(import backtest_v2|from backtest_v2)\b", re.M)
    for p in ROOT.glob("*.py"):
        # v2_experiments.py is the research engine built on v2 (2026-10-15) - research, not
        # production, and test_v2_experiments.py keeps production from importing it in turn
        if p.name in ("backtest_v2.py", "v2_experiments.py", "conftest.py"):
            continue
        assert not pat.search(p.read_text(encoding="utf-8", errors="replace")), p.name
    for p in (ROOT / "scripts").rglob("*.py"):
        assert not pat.search(p.read_text(encoding="utf-8", errors="replace")), str(p)


def test_a_dotted_ticker_in_the_cache_is_still_found():
    inp = _inputs(("AAA",))
    rows = inp.daily["AAA"].reset_index().assign(ticker="BRK.B")
    inp2 = v2.Inputs(rows, pd.DataFrame({"ticker": ["^SP500TR", "^IRX"], "date": [pd.Timestamp("2024-01-02")] * 2,
                                         "close": [1.0, 5.0]}),
                     pd.DataFrame(columns=["ticker", "date", "ratio"]), inp.empty_facts, {})
    assert "BRK-B" in inp2.daily and "BRK.B" not in inp2.daily


def test_a_month_without_a_cached_universe_is_skipped_and_said(monkeypatch, tmp_path):
    """Never today's list for a past month: the month is recorded as skipped instead."""
    def no_universe(as_of, allow_network=False):
        raise v2.uh.UniverseHistoryError("no cached universe for " + as_of)
    monkeypatch.setattr(v2.uh, "members_on", no_universe)
    monkeypatch.setattr(v2.Inputs, "load", classmethod(lambda cls, root=v2.ROOT: _inputs()))
    monkeypatch.setattr(v2, "CACHE", tmp_path)
    monkeypatch.setattr(v2, "SCORES_PATH", tmp_path / "s.parquet")
    monkeypatch.setattr(v2, "CENSUS_PATH", tmp_path / "c.json")
    monkeypatch.setattr(sys, "argv", ["backtest_v2.py", "--start", "2024-06-30", "--end", "2024-06-30"])
    v2.main()
    import json
    census = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert census[0]["month"] == "2024-06-30" and census[0]["skipped"].startswith("no cached universe")


def test_a_reused_symbol_is_never_scored_as_todays_company(monkeypatch):
    """2026-10-15: in 2012-2019 a symbol can have belonged to another company. Where that month's
    revision lists a CIK that differs from today's company under the symbol, today's company is not
    scored in its place - the member is found under its own current symbol, or counted."""
    class Snap:
        tickers = ("AAA",)
        n_members = 1
        staleness_days = 0
    monkeypatch.setattr(v2.uh, "members_on", lambda as_of, allow_network=False: Snap)
    monkeypatch.setattr(v2, "metrics_month", lambda recs, cfg, mr, rf: pd.DataFrame(
        {"Ticker": [r["Ticker"] for r in recs]}))
    monkeypatch.setattr(v2, "rank_month", lambda m, cfg, st: m.assign(Composite=50.0, Rank=1))
    monkeypatch.setattr(v2, "save_metrics", lambda *a, **k: None)
    reused = _inputs(("AAA",), ciks={"AAA": 7}, month_ciks={"2024-06-28": {"AAA": 99}})
    _, census = v2.build_month("2024-06-28", reused, v2.v2_config(CFG))
    assert census["built"] == 0 and census["not_scoreable"] == {"symbol_reused": ["AAA"]}
    same = _inputs(("AAA",), ciks={"AAA": 7}, month_ciks={"2024-06-28": {"AAA": 7}})
    _, census = v2.build_month("2024-06-28", same, v2.v2_config(CFG))
    assert census["built"] == 1 and census["renamed"] == {}
