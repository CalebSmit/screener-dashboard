"""Our own implied volatilities (implied_vol, 2026-11-24): the inversion, its inputs, and the
paired record it adds beside the source field.

Decided in research/2026-11-03-open-source-quant-stack.md section 1d (ADOPTED 2026-11-04); built
from its build order. What these tests pin, on inputs with known answers:

* the inversion recovers a volatility it priced with, with and without a dividend, and agrees with
  an independently written Cox-Ross-Rubinstein tree (the measurement's own cross-check);
* the dividend schedule is the stated rule, and a payer with no ex-date is "unknown", never "none";
* the Treasury curve is FRED's, interpolated in time, and a stale one is not used;
* every reading carries both constructions where it can, the main columns hold the published one
  (falling back, stamped), and the page shows only the published one.
"""
from __future__ import annotations

import importlib.util
import json
from collections import namedtuple
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import context_signals as cs
import generate_dashboard as gd
import implied_vol as iv

ROOT = Path(__file__).resolve().parents[1]
ql = pytest.importorskip("QuantLib")

TODAY = date(2026, 11, 3)
EXPIRY = date(2026, 12, 18)


def _ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


# --------------------------------------------------------------------------- the inversion
@pytest.mark.parametrize("divs", [[], [(date(2026, 11, 20), 1.0)]])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_the_inversion_recovers_the_volatility_it_priced_with(kind, divs):
    p = iv.american_price(kind, 100.0, 100.0, 0.25, 0.04, divs, TODAY, EXPIRY)
    got = iv.american_iv(kind, 100.0, 100.0, p, 0.04, divs, TODAY, EXPIRY)
    assert got == pytest.approx(0.25, abs=1e-3)


def test_the_inversion_agrees_with_an_independently_written_tree():
    """The measurement's cross-check (section 1d (b)): its CRR tree, separate code from QuantLib,
    on a dividend-carrying put and call. Half a volatility point is the bar it was held to."""
    spec = importlib.util.spec_from_file_location(
        "iv1103", ROOT / "research" / "measurements" / "2026-11-03-implied-vol-quantlib-vs-yahoo.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    divs = [(date(2026, 11, 20), 0.9)]
    for kind, K, price in (("call", 105.0, 2.10), ("put", 95.0, 1.75)):
        ours = iv.american_iv(kind, 101.0, K, price, 0.04, divs, TODAY, EXPIRY)
        theirs = m.crr_iv(kind, 101.0, K, price, 0.04, divs, TODAY, EXPIRY)
        assert ours is not None and theirs is not None
        assert abs(ours - theirs) < 0.005, (kind, ours, theirs)


def test_a_price_no_volatility_can_produce_gives_none():
    # below intrinsic value: a put struck 20 above the price for 5 cents
    assert iv.american_iv("put", 100.0, 120.0, 0.05, 0.04, [], TODAY, EXPIRY) is None
    assert iv.american_iv("call", 100.0, 100.0, 0.0, 0.04, [], TODAY, EXPIRY) is None


# --------------------------------------------------------------------------- the dividends
def test_a_stock_that_pays_nothing_has_an_empty_schedule():
    assert iv.dividend_schedule({"trailingAnnualDividendRate": 0.0}, TODAY, EXPIRY) == []


def test_a_payer_with_no_ex_date_is_unknown_not_none():
    """Pricing a payer's option as if no dividend fell in its life would bias the put leg; the
    reading keeps the source field instead."""
    assert iv.dividend_schedule({"dividendRate": 2.0}, TODAY, EXPIRY) is None
    assert iv.dividend_schedule({}, TODAY, EXPIRY) is None


def test_a_quarterly_payer_is_projected_from_its_last_ex_date():
    # AAPL-like: 1.08 a year, last 0.27 paid ex 2026-08-11 -> quarterly, next 2026-11-10
    rec = {"dividendRate": 1.08, "lastDividendValue": 0.27, "exDividendDate": _ts(date(2026, 8, 11))}
    sched = iv.dividend_schedule(rec, TODAY, date(2027, 2, 15))
    assert [d for d, _ in sched] == [date(2026, 11, 10), date(2027, 2, 9)]
    assert all(a == pytest.approx(0.27) for _, a in sched)


def test_an_announced_ex_date_is_the_first_payment():
    rec = {"dividendRate": 2.12, "lastDividendValue": 0.53, "exDividendDate": _ts(date(2026, 12, 1))}
    assert iv.dividend_schedule(rec, TODAY, EXPIRY) == [(date(2026, 12, 1), pytest.approx(0.53))]


def test_a_monthly_payer_is_monthly():
    # O-like: 3.29 a year, 0.272 a payment -> 12 a year
    rec = {"dividendRate": 3.29, "lastDividendValue": 0.272, "exDividendDate": _ts(date(2026, 10, 30))}
    sched = iv.dividend_schedule(rec, TODAY, date(2027, 1, 15))
    assert [d for d, _ in sched] == [date(2026, 11, 29), date(2026, 12, 29)]       # 30 days apart
    assert sched[0][1] == pytest.approx(3.29 / 12)


def test_the_trailing_rate_stands_in_when_the_forward_one_is_missing():
    rec = {"dividendRate": float("nan"), "trailingAnnualDividendRate": 2.0,
           "exDividendDate": _ts(date(2026, 12, 1))}
    assert iv.dividend_schedule(rec, TODAY, EXPIRY) == [(date(2026, 12, 1), pytest.approx(0.5))]


def test_an_ex_date_today_has_already_happened():
    rec = {"dividendRate": 4.0, "lastDividendValue": 1.0, "exDividendDate": _ts(TODAY)}
    assert iv.dividend_schedule(rec, TODAY, EXPIRY) == []          # next one is past the expiry


# --------------------------------------------------------------------------- the rate
def _curve_doc(obs: str, pcts=(4.0, 4.1, 4.2, 4.3)) -> dict:
    return {"points": [{"sid": s, "days": d, "date": obs, "pct": p}
                       for (s, d), p in zip(iv.CURVE_SERIES, pcts)]}


def test_the_curve_is_continuously_compounded_and_interpolated_in_time():
    curve = iv.curve_from(_curve_doc("2026-11-02"), TODAY)
    assert curve[0] == (30, pytest.approx(np.log(1.04)))
    assert iv.rate_for(curve, 60) == pytest.approx(np.log(1.04) + (np.log(1.041) - np.log(1.04)) * 30 / 61)
    assert iv.rate_for(curve, 5) == pytest.approx(np.log(1.04))              # flat beyond the ends


def test_a_stale_or_incomplete_curve_is_not_used():
    assert iv.curve_from(_curve_doc("2026-10-01"), TODAY) is None
    doc = _curve_doc("2026-11-02")
    doc["points"].pop()
    assert iv.curve_from(doc, TODAY) is None
    assert iv.curve_from(None, TODAY) is None


def test_a_failed_fetch_falls_back_to_the_cached_curve(tmp_path):
    path = tmp_path / "treasury_curve.json"
    path.write_text(json.dumps(_curve_doc("2026-11-02")), encoding="utf-8")

    def boom(today):
        raise OSError("no network")
    # mtime far in the past makes the cache "not fresh", so a fetch is tried and fails
    curve = iv.treasury_curve(TODAY, path=path, fetch=boom, now=path.stat().st_mtime + 10 ** 7)
    assert curve is not None and len(curve) == 4
    assert iv.treasury_curve(TODAY, path=tmp_path / "none.json", fetch=boom) is None


def test_the_curve_is_cached_where_the_refresh_may_write():
    """The 20:00 refresh writes only data/options/ (settled row -1)."""
    assert iv.CURVE_PATH.parent == ROOT / "data" / "options"


def test_no_inverter_without_its_inputs():
    curve = iv.curve_from(_curve_doc("2026-11-02"), TODAY)
    assert iv.inverter({"dividendRate": 2.0}, curve, TODAY, EXPIRY) is None    # payer, no ex-date
    assert iv.inverter({"trailingAnnualDividendRate": 0.0}, None, TODAY, EXPIRY) is None
    assert callable(iv.inverter({"trailingAnnualDividendRate": 0.0}, curve, TODAY, EXPIRY))


# --------------------------------------------------------------------------- the reading
SPOT, SIGMA, R = 100.0, 0.30, 0.04
STRIKES = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0]


def _priced_chain(source_iv=0.20):
    """A chain whose mids are QuantLib prices at 30% - and whose source field says 20%."""
    rows = {}
    for kind in ("call", "put"):
        mids = [iv.american_price(kind, SPOT, k, SIGMA, R, [], TODAY, EXPIRY) for k in STRIKES]
        rows[kind] = pd.DataFrame({"strike": STRIKES, "bid": [m * 0.99 for m in mids],
                                   "ask": [m * 1.01 for m in mids],
                                   "impliedVolatility": [source_iv] * len(STRIKES),
                                   "openInterest": [100] * len(STRIKES)})
    return rows["call"], rows["put"]


def _invert(kind, S, K, price):
    return iv.american_iv(kind, S, K, price, R, [], TODAY, EXPIRY)


def test_a_reading_carries_both_constructions():
    calls, puts = _priced_chain()
    out = cs.options_context(calls, puts, SPOT, EXPIRY.isoformat(), TODAY, False, invert=_invert)
    assert out["_ctx_opt_own_iv"] == pytest.approx(SIGMA, abs=0.005)
    assert out["_ctx_opt_own_skew_put_iv"] == pytest.approx(SIGMA, abs=0.005)
    assert out["_ctx_opt_own_skew_strike"] == 95.0 or out["_ctx_opt_own_skew_strike"] == 90.0
    assert out["_ctx_opt_own_skew"] == pytest.approx(0.0, abs=0.01)    # flat smile priced in
    assert out["_ctx_opt_src_iv"] == pytest.approx(0.20)
    assert out["_ctx_opt_src_skew"] == pytest.approx(0.0)


def test_the_main_columns_hold_the_published_construction(monkeypatch):
    calls, puts = _priced_chain()
    # Ours is published since 2026-11-26, after the whole-universe coverage count.
    assert cs.PUBLISHED_IV_CONSTRUCTION == "own_mid"
    monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", "source")
    out = cs.options_context(calls, puts, SPOT, EXPIRY.isoformat(), TODAY, False, invert=_invert)
    assert out["_ctx_opt_iv_src"] == "source" and out["_ctx_opt_iv"] == pytest.approx(0.20)
    monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", "own_mid")
    out = cs.options_context(calls, puts, SPOT, EXPIRY.isoformat(), TODAY, False, invert=_invert)
    assert out["_ctx_opt_iv_src"] == "own_mid" and out["_ctx_opt_iv"] == pytest.approx(SIGMA, abs=0.005)
    assert out["_ctx_opt_skew_put_iv"] == out["_ctx_opt_own_skew_put_iv"]
    assert out["_ctx_opt_skew"] == pytest.approx(out["_ctx_opt_skew_put_iv"] - out["_ctx_opt_skew_call_iv"])


def test_a_missing_construction_falls_back_to_the_other_stamped(monkeypatch):
    calls, puts = _priced_chain(source_iv=float("nan"))
    out = cs.options_context(calls, puts, SPOT, EXPIRY.isoformat(), TODAY, False, invert=_invert)
    assert out["_ctx_opt_iv_src"] == "own_mid"                  # the source has none
    assert "_ctx_opt_src_iv" not in out
    monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", "own_mid")
    calls, puts = _priced_chain()
    out = cs.options_context(calls, puts, SPOT, EXPIRY.isoformat(), TODAY, False, invert=None)
    assert out["_ctx_opt_iv_src"] == "source" and not any(k.startswith("_ctx_opt_own_") for k in out)


def test_without_an_inverter_a_reading_is_what_it_always_was():
    calls, puts = _priced_chain()
    out = cs.options_context(calls, puts, SPOT, EXPIRY.isoformat(), TODAY, False)
    assert out["_ctx_opt_iv"] == pytest.approx(0.20) and out["_ctx_opt_iv_src"] == "source"
    assert not any(k.startswith("_ctx_opt_own_") for k in out)


class _FakeTicker:
    options = (EXPIRY.isoformat(),)

    def option_chain(self, exp):
        calls, puts = _priced_chain()
        return namedtuple("Options", "calls puts underlying")(calls, puts, {"regularMarketPrice": SPOT})


@pytest.mark.parametrize("path", ["refresh", "live"])
def test_both_option_paths_add_our_own_volatilities(monkeypatch, path):
    import yfinance

    import context_fetch
    import options_cache
    monkeypatch.setattr(yfinance, "Ticker", lambda t: _FakeTicker())
    monkeypatch.setattr(iv, "rate_for", lambda curve, days: R)
    fetch = options_cache._fetch_one if path == "refresh" else context_fetch._options_for
    curve = iv.curve_from(_curve_doc("2026-11-02"), TODAY)
    out = fetch("ZZZ", SPOT, None, TODAY, rec={"trailingAnnualDividendRate": 0.0}, curve=curve)
    assert out["_ctx_opt_own_iv"] == pytest.approx(SIGMA, abs=0.005)
    assert out["_ctx_opt_iv_src"] == cs.PUBLISHED_IV_CONSTRUCTION
    # a payer the fetch could not date: the source field only
    out = fetch("ZZZ", SPOT, None, TODAY, rec={"dividendRate": 1.0}, curve=curve)
    assert out["_ctx_opt_iv_src"] == "source" and "_ctx_opt_own_iv" not in out


def test_the_fetch_keeps_the_dividend_fields_from_the_info_it_already_pulls():
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    for f in ("exDividendDate", "lastDividendValue"):
        assert f'rec["{f}"]' in src and f'_safe(info, "{f}")' in src
    import options_cache
    assert {"exDividendDate", "lastDividendValue", "dividendRate"} <= set(options_cache.DIVIDEND_FIELDS)


def test_the_refresh_reads_the_dividend_fields_when_the_run_has_them(tmp_path):
    import options_cache
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    pd.DataFrame({"Ticker": ["A"], "price_latest": [10.0], "earningsTimestampStart": [None],
                  "dividendRate": [1.0], "exDividendDate": [_ts(date(2026, 12, 1))],
                  "lastDividendValue": [0.25]}).to_parquet(run / "00_raw_fetch.parquet")
    rec = options_cache._records_from_latest_run(tmp_path)[0]
    assert rec["exDividendDate"] == _ts(date(2026, 12, 1)) and rec["lastDividendValue"] == 0.25
    # an older run without them still loads
    pd.DataFrame({"Ticker": ["A"], "price_latest": [10.0], "earningsTimestampStart": [None]}).to_parquet(
        run / "00_raw_fetch.parquet")
    assert "exDividendDate" not in options_cache._records_from_latest_run(tmp_path)[0]


# --------------------------------------------------------------------------- the page
def _row(**over):
    row = {"_ctx_opt_status": "ok", "_ctx_opt_move": 0.07, "_ctx_opt_straddle": 7.3,
           "_ctx_opt_atm_strike": 100.0, "_ctx_opt_spot": 100.0, "_ctx_opt_spot_src": "chain",
           "_ctx_opt_iv": 0.3, "_ctx_opt_skew": 0.04, "_ctx_opt_skew_put_iv": 0.34,
           "_ctx_opt_skew_strike": 95.0, "_ctx_opt_skew_call_iv": 0.30, "_ctx_opt_skew_call_strike": 100.0,
           "_ctx_opt_quote_date": "2026-10-20"}
    row.update(over)
    return {k: v for k, v in row.items() if v is not None}


@pytest.mark.parametrize("stamp, published, shown", [
    (None, "source", True),             # unstamped is a source reading
    ("source", "source", True),
    ("own_mid", "source", False),       # the record would refuse it, so the page does too
    ("own_mid", "own_mid", True),
    ("source", "own_mid", False),
    (None, "own_mid", False),
])
def test_the_page_shows_only_the_construction_the_record_counts(monkeypatch, stamp, published, shown):
    monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", published)
    out = gd._stock_context(_row(_ctx_opt_iv_src=stamp), "ZZZ", None)
    assert ("oi" in out) is shown and ("osk" in out) is shown and ("oskp" in out) is shown
    assert out.get("ois") == (published if shown else None)
    assert bool(out.get("oniv")) is (not shown)
    assert out.get("oniv") == (None if shown else published)   # names what the panel does report
    assert "om" in out                                         # the expected move does not depend on it


def test_the_page_and_the_record_read_one_predicate(monkeypatch):
    import context_eval as ce
    log = pd.DataFrame({"_ctx_opt_iv_src": ["source", "own_mid", None]})
    for published in cs.IV_CONSTRUCTIONS:
        monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", published)
        counted = [bool(x) for x in ce.iv_is_the_published_construction(log)]
        shown = ["oi" in gd._stock_context(_row(_ctx_opt_iv_src=s), "ZZZ", None) for s in log["_ctx_opt_iv_src"]]
        assert counted == shown


def test_the_panel_says_how_its_volatilities_were_worked_out():
    js = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert "c.ois === 'own_mid'" in js
    assert "worked out here from each option&rsquo;s bid&ndash;ask midpoint" in js
    assert "the data source&rsquo;s own figures; it does not publish how it computes them" in js
    assert "c.oniv" in js
    # 2026-11-26: under own_mid a withheld reading is one ours could not be worked out for, and the
    # panel says that rather than "a different way"; a reading with our IV and no skew says why.
    assert "c.oniv === 'own_mid'" in js
    assert "they could not be worked out for this reading" in js
    assert "No put skew this reading: its two options could not both be worked out" in js


# --------------------------------------------------------------------------- the cache across a flip
def _cached_source_reading(**over):
    """A reading the 20:00 refresh cached while ``source`` was published: main columns are the
    source's, our own figures sit beside them (the 2026-11-25 cache's shape)."""
    r = {"_ctx_opt_status": "ok", "_ctx_opt_move": 0.07, "_ctx_opt_straddle": 7.3,
         "_ctx_opt_atm_strike": 100.0, "_ctx_opt_spot": 100.0, "_ctx_opt_spot_src": "chain",
         "_ctx_opt_expiry": "2026-12-18", "_ctx_opt_days": 23, "_ctx_opt_quote_date": "2026-11-25",
         "_ctx_opt_iv_src": "source",
         "_ctx_opt_iv": 0.366, "_ctx_opt_skew": 0.05, "_ctx_opt_skew_put_iv": 0.41,
         "_ctx_opt_skew_strike": 95.0, "_ctx_opt_skew_call_iv": 0.36, "_ctx_opt_skew_call_strike": 100.0,
         "_ctx_opt_skew_in_window": True,
         "_ctx_opt_src_iv": 0.366, "_ctx_opt_src_skew": 0.05,
         "_ctx_opt_own_iv": 0.340, "_ctx_opt_own_call_iv": 0.345, "_ctx_opt_own_put_iv": 0.335,
         "_ctx_opt_own_skew": 0.013, "_ctx_opt_own_skew_put_iv": 0.358, "_ctx_opt_own_skew_strike": 95.0,
         "_ctx_opt_own_skew_call_iv": 0.345, "_ctx_opt_own_skew_call_strike": 100.0}
    r.update(over)
    return {k: v for k, v in r.items() if v is not None}


def test_a_cached_reading_is_reselected_for_the_published_construction(monkeypatch):
    monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", "own_mid")
    out = cs.in_published_construction(_cached_source_reading())
    assert out["_ctx_opt_iv_src"] == "own_mid"
    assert out["_ctx_opt_iv"] == 0.340 and out["_ctx_opt_skew"] == 0.013
    assert out["_ctx_opt_skew_put_iv"] == 0.358 and out["_ctx_opt_skew_call_iv"] == 0.345
    assert out["_ctx_opt_skew_strike"] == 95.0 and out["_ctx_opt_skew_in_window"] is True
    assert out["_ctx_opt_status"] == "ok" and out["_ctx_opt_move"] == 0.07     # the move is untouched
    assert out["_ctx_opt_src_iv"] == 0.366                                    # the pair is kept
    # and the page shows it, under our construction's description
    page = gd._stock_context(out, "ZZZ", None)
    assert page["oi"] == pytest.approx(0.340) and page["ois"] == "own_mid" and "oniv" not in page


def test_a_reading_with_our_iv_and_no_skew_drops_the_sources_skew(monkeypatch):
    monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", "own_mid")
    r = _cached_source_reading(**{k: None for k in ("_ctx_opt_own_skew", "_ctx_opt_own_skew_put_iv",
                                                     "_ctx_opt_own_skew_strike", "_ctx_opt_own_skew_call_iv",
                                                     "_ctx_opt_own_skew_call_strike")})
    out = cs.in_published_construction(r)
    assert out["_ctx_opt_iv"] == 0.340 and out["_ctx_opt_iv_src"] == "own_mid"
    assert not any(k in out for k in ("_ctx_opt_skew", "_ctx_opt_skew_put_iv", "_ctx_opt_skew_call_iv",
                                      "_ctx_opt_skew_in_window"))      # never a source skew under our stamp


def test_a_reading_with_none_of_ours_stays_stamped_and_withheld(monkeypatch):
    monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", "own_mid")
    r = _cached_source_reading(_ctx_opt_status="partial", _ctx_opt_move=None, _ctx_opt_straddle=None,
                               **{k: None for k in cs.OWN_KEYS})
    out = cs.in_published_construction(r)
    assert out == r and out["_ctx_opt_iv_src"] == "source"
    page = gd._stock_context(out, "ZZZ", None)
    assert "oi" not in page and page["oniv"] == "own_mid"


def test_reselection_is_a_no_op_in_the_published_construction(monkeypatch):
    for published in cs.IV_CONSTRUCTIONS:
        monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", published)
        r = _cached_source_reading(_ctx_opt_iv_src=published)
        assert cs.in_published_construction(r) is r
    # back to source: the ATM figure returns, and no skew - its legs were never kept in the pair
    monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", "source")
    out = cs.in_published_construction(_cached_source_reading(_ctx_opt_iv_src="own_mid"))
    assert out["_ctx_opt_iv"] == 0.366 and out["_ctx_opt_iv_src"] == "source" and "_ctx_opt_skew" not in out


def test_the_cache_serves_readings_in_the_published_construction(monkeypatch):
    import options_cache
    monkeypatch.setattr(cs, "PUBLISHED_IV_CONSTRUCTION", "own_mid")
    cache = {"quote_date": "2026-11-25", "readings": {"ZZZ": _cached_source_reading()}}
    out = options_cache.reading_for(cache, "ZZZ", date(2026, 11, 27))
    assert out["_ctx_opt_iv_src"] == "own_mid" and out["_ctx_opt_iv"] == 0.340
    assert out["_ctx_opt_days"] == 21
