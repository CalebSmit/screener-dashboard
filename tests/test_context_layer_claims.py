"""The context layer's remaining computational sentences, checked against the code that makes them true.

WHY THIS EXISTS - 2026-10-20, the close-out of CLAUDE.md priority 0.11 (``plan/context-layer.md``).
An audit of every sentence the context panels print that says how a number is computed found
five that had no registered claim, and three defects behind them:

* **A live false statement.** A reading fetched live carried no stamp, and the options panel's
  fallback said "Quotes are the previous close." The 13:00 session's live run published at 13:21
  CT on 2026-10-20 with quotes taken mid-session, so the live page told readers of 485 of 501
  stocks that mid-session quotes were the previous close. A live reading now records its session,
  its Eastern clock time and whether the market was open (``options_cache.live_quote_stamp``).
* **The evidence record took a mid-session run's prices as closes.** The same run replaced the data
  loop's log for 2026-10-20 with 502 prices dated 2026-10-20 - the unfinished bar - while every
  other log holds the previous session's closes. ``context_signals.mid_session_reason`` now keeps
  such a run out of the log.
* **The evaluation spliced two constructions of the put skew.** ``plan/context-layer.md`` item 4
  says the readings before Xing, Zhang & Zhao's construction are excluded, not spliced; the
  harness evaluated them all. It now uses the generator's own rule (a skew is the paper's only
  with both legs recorded).

Plus two sentences that said more than the code: the Recent move note said "the screener's own
momentum signal skips the latest month" (a quarter of Momentum, Jensen's alpha, does not), and the
Market panel said "the stat strip shows whether the cut is in force" (it shows a card only when it
is). Each claim here is checked on what the code *does* - a browser render of the generator's own
output, or the function on an input with a known answer - not on the presence of a string, because
"a claim whose check tests the plumbing instead of the assertion is an unchecked claim" (CLAUDE.md,
0.11 item 2).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import claims  # noqa: E402
import context_eval as ce  # noqa: E402
import context_fetch  # noqa: E402
import context_signals as cs  # noqa: E402
import generate_dashboard as gd  # noqa: E402
import market_context as mc  # noqa: E402
import options_cache  # noqa: E402
import stock_summary  # noqa: E402
from factor_engine import compute_metrics  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

ET = timezone(timedelta(hours=-4))          # EDT, fixed so the cases do not depend on the tz database
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _payload(name: str) -> dict:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"no published {name}")
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


# --------------------------------------------------------------------------- when the quotes are from
@pytest.mark.parametrize("instant, session, clock, intraday", [
    (datetime(2026, 10, 20, 14, 21, tzinfo=ET), "2026-10-20", "14:21", True),    # the 13:21 CT run
    (datetime(2026, 10, 20, 9, 30, tzinfo=ET), "2026-10-20", "09:30", True),     # the open
    (datetime(2026, 10, 20, 16, 0, tzinfo=ET), "2026-10-20", "16:00", False),    # the close
    (datetime(2026, 10, 20, 21, 1, tzinfo=ET), "2026-10-20", "21:01", False),    # the evening refresh
    (datetime(2026, 10, 20, 3, 0, tzinfo=ET), "2026-10-19", "03:00", False),     # before the open
    (datetime(2026, 10, 19, 8, 0, tzinfo=ET), "2026-10-16", "08:00", False),     # Monday pre-open -> Friday
    (datetime(2026, 10, 18, 12, 0, tzinfo=ET), "2026-10-16", "12:00", False),    # Sunday -> Friday
])
def test_a_live_reading_names_its_session_and_clock_time(instant, session, clock, intraday):
    stamp = options_cache.live_quote_stamp(instant)
    assert stamp == {"_ctx_opt_quote_date": session, "_ctx_opt_quote_time": clock,
                     "_ctx_opt_quote_intraday": intraday}
    assert options_cache.is_session_date(stamp["_ctx_opt_quote_date"])


def test_the_context_pass_stamps_a_live_reading(monkeypatch):
    monkeypatch.setattr(context_fetch, "_context_for",
                        lambda t, p, e, d, with_options=True:
                        {"_ctx_opt_status": "ok", "_ctx_opt_move": 0.05, "_ctx_insider": "[]"})
    raw = [{"Ticker": "AAA", "price_latest": 10.0}]
    context_fetch.enrich(raw, log=lambda *_a: None, options_live=True,
                         now=lambda: datetime(2026, 10, 20, 14, 21, tzinfo=ET))
    assert raw[0]["_ctx_opt_quote_date"] == "2026-10-20"
    assert raw[0]["_ctx_opt_quote_time"] == "14:21"
    assert raw[0]["_ctx_opt_quote_intraday"] is True


def test_a_refresh_run_during_the_session_is_marked_mid_session(tmp_path):
    """The 2026-10-09 smoke test ran the refresh at 09:46 ET; such a cache must not read as a close."""
    options_cache.refresh([{"Ticker": "AAA", "price_latest": 10.0}], root=tmp_path,
                          log=lambda *_a: None, probe=lambda _t: "ok",
                          fetch=lambda *a, **k: {"_ctx_opt_status": "ok", "_ctx_opt_expiry": "2026-12-18"},
                          now=datetime(2026, 10, 9, 9, 46, tzinfo=ET))
    r = options_cache.load(tmp_path)["readings"]["AAA"]
    assert r["_ctx_opt_quote_intraday"] is True and r["_ctx_opt_quote_time"] == "09:46"
    served = options_cache.reading_for(options_cache.load(tmp_path), "AAA", datetime(2026, 10, 9).date())
    assert served["_ctx_opt_quote_intraday"] is True, "the flag must reach the page through the cache"


def test_the_quote_time_reaches_the_payload():
    assert gd.CTX_FIELDS["oqt"] == "_ctx_opt_quote_time"
    assert gd.CTX_FIELDS["oqi"] == "_ctx_opt_quote_intraday"
    out = gd._stock_context({"_ctx_opt_status": "ok", "_ctx_opt_quote_date": "2026-10-20",
                             "_ctx_opt_quote_time": "14:21", "_ctx_opt_quote_intraday": True},
                            "AAA", None)
    assert out["oqd"] == "2026-10-20" and out["oqt"] == "14:21" and out["oqi"] is True


# --------------------------------------------------------------------------- the page, rendered
@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The generator's own page (not the committed file) with the published payloads beside it."""
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"no published {f}")
    d = tmp_path_factory.mktemp("page")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof renderContext === 'function' && typeof D !== 'undefined' && "
                             "Object.values(D.stock_detail).some(s => s.ctx)", timeout=60000)
        yield pg
        b.close()
        assert not errors, errors


def _options_text(page, extra: dict) -> str:
    c = {"os": "ok", "om": 0.05, "ox": "2026-11-20", "od": 31, "px": 100.0, "oi": 0.30}
    c.update(extra)
    s = {"sector": "Energy", "raw": {}, "ctx": c}
    page.evaluate("s => renderContext('ZZZ', s)", s)
    return re.sub(r"\s+", " ", page.evaluate("document.getElementById('modal-context').textContent"))


def test_the_page_says_when_each_kind_of_quote_was_taken(page):
    mid = _options_text(page, {"oqd": "2026-10-20", "oqt": "14:21", "oqi": True})
    assert "Quotes were taken at 14:21 ET on Oct 20, while the market was open" in mid
    assert "close of" not in mid and "previous close" not in mid

    evening = _options_text(page, {"oqd": "2026-10-20", "oqt": "21:01", "oqi": False})
    assert "Quotes are the close of Oct 20, collected at 21:01 ET." in evening

    cached = _options_text(page, {"oqd": "2026-10-19"})
    assert "Quotes are the close of Oct 19; options are not quoted at the hour this run fetches" in cached

    unknown = _options_text(page, {})
    assert "did not record when its option quotes were taken" in unknown
    for text in (mid, evening, cached, unknown):
        assert not claims.forbidden_hits(text)
        assert not stock_summary.advice_terms_in(text)


def test_the_page_trend_word_is_the_engines(page):
    """One rule for the chip, the teaser and the Context filter: the page's trendOf is
    context_signals.trend_state, case for case, including the ties."""
    word = {"uptrend": "up", "downtrend": "down", "mixed": "mixed", None: None}
    for px in (90.0, 100.0, 110.0):
        for s50 in (None, 95.0, 100.0, 105.0):
            ctx = {"_ctx_last_close": px, "_ctx_sma50": s50, "_ctx_sma200": 100.0}
            js = page.evaluate("c => trendOf(c)", {"px": px, "s50": s50, "s200": 100.0})
            assert js == word[cs.trend_state(ctx)], (px, s50)


def test_the_sector_median_on_the_page_is_the_median_of_the_published_moves(page):
    data, ctx = _payload("dashboard_data.js"), _payload("dashboard_context.js")
    by: dict[str, list] = {}
    for t, s in data["stock_detail"].items():
        r = (ctx["ctx"].get(t) or {}).get("r1m")
        if r is not None:
            by.setdefault(s.get("sector"), []).append(r)
    assert len(by) >= 10
    for sector, vals in by.items():
        js = page.evaluate("s => { SECTOR_R1M = null; return sectorR1m(s); }", sector)
        assert js == pytest.approx(float(np.median(vals)), abs=1e-12), sector


def test_the_stat_strip_shows_a_momentum_cut(page):
    cut = page.evaluate("""() => { const w = D.weights; const keep = JSON.stringify(w);
        w.factor_weights = Object.assign({}, w.factor_weights, {momentum: 9.1});
        w.base_factor_weights = Object.assign({}, w.factor_weights, {momentum: 13});
        w.factor_weights_adjusted = true; renderKPIs();
        const t = document.getElementById('kpi-row').textContent;
        D.weights = JSON.parse(keep); renderKPIs(); return t; }""")
    assert "Momentum weight" in cut and "cut from" in cut
    none = page.evaluate("""() => { const w = D.weights; const keep = JSON.stringify(w);
        w.factor_weights_adjusted = false; renderKPIs();
        const t = document.getElementById('kpi-row').textContent;
        D.weights = JSON.parse(keep); renderKPIs(); return t; }""")
    assert "Momentum weight" not in none


# --------------------------------------------------------------------------- price windows
def test_the_windows_on_the_page_are_the_windows_in_the_code():
    assert (cs.SMA_SHORT, cs.SMA_LONG, cs.SLOPE_LAG, cs.VOL_SHORT, cs.VOL_LONG) == (50, 200, 20, 20, 63)
    idx = pd.bdate_range(end="2026-10-07", periods=300)
    c = pd.Series(np.exp(np.linspace(0, 0.6, 300)), index=idx)
    v = pd.Series(np.r_[np.full(280, 100.0), np.full(20, 300.0)], index=idx)
    out = cs.price_context(c, v)
    for label, days in (("5d", 7), ("1m", 30), ("3m", 91)):
        prior = c.loc[c.index <= c.index[-1] - pd.Timedelta(days=days)].iloc[-1]
        assert out[f"_ctx_ret_{label}"] == pytest.approx(c.iloc[-1] / prior - 1)
    assert out["_ctx_vol_ratio"] == pytest.approx(300.0 / ((43 * 100 + 20 * 300) / 63))
    js = gd._js_context()
    for label in ("5 days", "1 month", "3 months", "Volume, 20 days vs 3 months", "vs 200-day average",
                  "vs 50-day average", "From 52-week high", "rising' : 'falling') + ' over the last month"):
        assert label in js, label


# --------------------------------------------------------------------------- momentum skips a month
def _rec(**over):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_metrics import _make_rec
    return _make_rec(**over)


def test_the_skip_month_returns_ignore_the_latest_month():
    # Only the latest price differs (the same split-adjusted series the engine reads, price_latest)
    a = compute_metrics([_rec(price_latest=100.0, price_1m_ago=98.0)], pd.Series(dtype=float)).iloc[0]
    b = compute_metrics([_rec(price_latest=130.0, price_1m_ago=98.0)], pd.Series(dtype=float)).iloc[0]
    assert a["return_12_1"] == pytest.approx(b["return_12_1"])
    assert a["return_6m"] == pytest.approx(b["return_6m"])
    # Jensen's alpha is built on the full twelve months, so the latest month moves its input
    assert a["return_12m"] != pytest.approx(b["return_12m"])


def test_the_momentum_sentence_matches_the_configured_weights():
    w = CFG["metric_weights"]["momentum"]
    total = sum(float(x) for x in w.values())
    skip = (float(w["return_12_1"]) + float(w["return_6m"])) / total
    rest = {k for k, x in w.items() if float(x) > 0} - {"return_12_1", "return_6m"}
    assert rest == {"jensens_alpha"}, rest
    assert skip == pytest.approx(0.75), "the sentence says three quarters - change it with the weights"
    js = gd._js_context()
    assert "12-1 and 6-1 month returns, three quarters of its Momentum category, skip the latest month" in js
    assert "Jensen&rsquo;s alpha, the other quarter, uses the full twelve months" in js
    assert not claims.forbidden_hits(js)


# --------------------------------------------------------------------------- the momentum rule, as the backdrop states it
def test_the_volatility_reading_describes_the_configured_rule():
    r = {x["k"]: x for x in mc.readings({"VIXCLS": {"last": 30.0, "median_10y": 17.0}}, None)}["vol"]["text"]
    assert "cuts momentum's weight in their panic state" in r and "never raises it" in r
    assert "down over two years and volatile" in r
    rule = CFG["momentum_regime"]
    assert int(rule["return_window_months"]) == 24      # "over two years"
    assert float(rule["vol_threshold"]) > 0             # "and volatile": a fixed volatility bar
    note = next(n for n in mc.FACTOR_NOTES if n["title"] == "Momentum and rebounds")["text"]
    assert "when the cut is in force, the stat strip at the top of the page says so" in note


def test_quality_is_one_of_the_two_heaviest_categories():
    fw = {k: float(v) for k, v in CFG["factor_weights"].items()}
    top_two = sorted(fw.values(), reverse=True)[1]
    assert fw["quality"] >= top_two
    note = next(n for n in mc.FACTOR_NOTES if n["title"] == "Quality in downturns")["text"]
    assert "one of this screener's two heaviest categories" in note


# --------------------------------------------------------------------------- the market cards
def test_the_ten_year_percentile_counts_half_of_ties():
    idx = pd.date_range("2014-01-01", "2026-10-01", freq="D")
    vals = np.where(idx < pd.Timestamp("2016-10-01"), 99.0, np.tile([1.0, 2.0, 3.0], len(idx) // 3 + 1)[:len(idx)])
    s = pd.Series(vals, index=idx)
    s.iloc[-1] = 2.0
    out = mc.summarise("DGS10", s)
    ten = s.loc[s.index >= s.index[-1] - pd.Timedelta(days=3653)]
    expected = (ten < 2.0).mean() * 100 + (ten == 2.0).mean() * 50
    assert out["pct_10y"] == pytest.approx(round(expected, 1))
    assert 45 < out["pct_10y"] < 55, "the 99s older than ten years must not count"


def test_changes_are_against_the_last_value_30_and_365_days_back():
    idx = pd.bdate_range("2015-01-01", "2026-10-01")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    out = mc.summarise("DGS10", s)
    last_d = s.index[-1]
    for key, days in (("chg_1m", 30), ("chg_1y", 365)):
        prev = s.loc[s.index <= last_d - pd.Timedelta(days=days)].iloc[-1]
        assert out[key] == pytest.approx(s.iloc[-1] - prev)


def test_the_vix_reading_quotes_the_ten_year_median():
    idx = pd.date_range("2015-01-01", "2026-10-01", freq="B")
    s = pd.Series(np.linspace(10, 40, len(idx)), index=idx)
    summ = {"VIXCLS": mc.summarise("VIXCLS", s)}
    ten = s.loc[s.index >= s.index[-1] - pd.Timedelta(days=3653)]
    assert summ["VIXCLS"]["median_10y"] == pytest.approx(round(float(ten.median()), 3))
    text = {r["k"]: r for r in mc.readings(summ, None)}["vol"]["text"]
    assert f"against a ten-year median of {ten.median():.1f}" in text


# --------------------------------------------------------------------------- rates
def test_the_share_of_swings_explained_is_the_squared_correlation():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2025-09-01", periods=250)
    dy = pd.Series(rng.normal(0, 0.06, len(idx)), index=idx)
    r = 0.3 * dy + rng.normal(0, 0.02, len(idx))
    out = cs.rate_sensitivity({d.strftime("%Y-%m-%d"): v for d, v in r.items()}, dy)
    assert out["r2"] == pytest.approx(np.corrcoef(r.values, dy.values)[0, 1] ** 2)


def test_sector_rate_bars_are_medians_of_five_or_more_stocks(monkeypatch):
    monkeypatch.setattr(mc, "load", lambda: {"series": {}})
    sd = {f"A{i}": {"sector": "A", "ctx": {"rb": v}} for i, v in enumerate([0.1, 0.5, -0.2, 0.3, 0.9])}
    sd.update({f"B{i}": {"sector": "B", "ctx": {"rb": 0.1}} for i in range(4)})
    m = gd._market_block(sd)
    assert m["sector_rate"] == {"A": {"median": 0.3, "n": 5}}


# --------------------------------------------------------------------------- the record
def _raw_with_every_published_column() -> pd.DataFrame:
    row = {"Ticker": "AAA"}
    for col in gd.CTX_FIELDS.values():
        row[col] = 0.5
    row.update({"_ctx_last_date": "2026-09-14", "_ctx_opt_status": "ok", "_ctx_opt_quote_date": "2026-09-14",
                "_ctx_opt_quote_time": "21:01", "_ctx_opt_quote_intraday": False,
                "_ctx_weekly": "{}", "_ctx_valhist": "{}", "_ctx_insider": "[]",
                "_ctx_vh_ey_pct": 40.0, "_ctx_vh_fy_pct": 60.0})
    return pd.DataFrame([row])


def test_every_number_the_panel_shows_is_logged(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    _raw_with_every_published_column().to_parquet(run / "00_raw_fetch.parquet")
    monkeypatch.setattr(cs, "CONTEXT_LOG_DIR", tmp_path / "log")
    assert cs.write_context_log(run, "2026-09-15") == 1
    log = pd.read_parquet(tmp_path / "log" / "2026-09-15.parquet")
    missing = [c for c in gd.CTX_FIELDS.values() if c not in log.columns]
    assert not missing, f"shown on the panel but not logged: {missing}"
    for c in ("_ctx_vh_ey_pct", "_ctx_vh_fy_pct", "_ctx_ins_buy_n", "_ctx_ins_buy_value", "_ctx_ins_sell_value"):
        assert c in log.columns, c


def _run(tmp_path, last_date: str, start: str) -> Path:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    pd.DataFrame({"Ticker": ["A", "B"], "_ctx_last_date": [last_date] * 2,
                  "_ctx_last_close": [10.0, 20.0]}).to_parquet(run / "00_raw_fetch.parquet")
    (run / "meta.json").write_text(json.dumps({"start_time": start}), encoding="utf-8")
    return run


def test_a_mid_session_run_does_not_replace_the_log(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CONTEXT_LOG_DIR", tmp_path / "log")
    before = _run(tmp_path, "2026-09-14", "2026-09-15T02:00:03-05:00")      # the data loop: Monday's closes
    assert cs.write_context_log(before, "2026-09-15") == 2
    first = (tmp_path / "log" / "2026-09-15.parquet").read_bytes()
    mid = _run(tmp_path, "2026-09-15", "2026-09-15T13:21:50-05:00")          # 14:21 ET, today's unfinished bar
    assert "unfinished bar" in (cs.mid_session_reason(mid, pd.read_parquet(mid / "00_raw_fetch.parquet")) or "")
    assert cs.write_context_log(mid, "2026-09-15") == 0
    assert (tmp_path / "log" / "2026-09-15.parquet").read_bytes() == first


def test_an_evening_run_still_records(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CONTEXT_LOG_DIR", tmp_path / "log")
    run = _run(tmp_path, "2026-09-15", "2026-09-15T20:27:00-05:00")          # 21:27 ET: a complete bar
    raw = pd.read_parquet(run / "00_raw_fetch.parquet")
    assert cs.mid_session_reason(run, raw, now=datetime(2026, 9, 15, 21, 50, tzinfo=ET)) is None
    assert cs.write_context_log(run, "2026-09-15") == 2


def test_the_committed_logs_hold_closes_only():
    logs = sorted((ROOT / "data" / "context_log").glob("*.parquet"))
    if not logs:
        pytest.skip("no committed context logs")
    for f in logs:
        d = pd.read_parquet(f)
        last = d["_ctx_last_date"].dropna().astype(str)
        assert (last <= f.stem).all(), f.name
        if "_ctx_opt_quote_intraday" in d.columns:
            assert not d["_ctx_opt_quote_intraday"].fillna(False).astype(bool).any(), f.name
    repaired = ROOT / "data" / "context_log" / "2026-10-20.parquet"
    if repaired.exists():
        assert pd.read_parquet(repaired)["_ctx_last_date"].dropna().mode().iloc[0] == "2026-10-19"


_PRICE_COLS = ("_ctx_sma200", "_ctx_sma50", "_ctx_ret_1m", "_ctx_ret_5d", "_ctx_ret_3m",
               "_ctx_high_52w", "_ctx_low_52w", "_ctx_vol_ratio")


def _log(**cols) -> pd.DataFrame:
    """A context log with the price columns every real log carries."""
    n = len(next(iter(cols.values())))
    return pd.DataFrame({**{c: [1.0] * n for c in _PRICE_COLS}, **cols})


def test_the_harness_excludes_skews_without_legs():
    log = _log(**{"Ticker": ["A", "B", "C"], "_ctx_last_close": [1.0, 2.0, 3.0],
                        "_ctx_opt_status": ["ok", "ok", "ok"], "_ctx_opt_skew": [0.05, 0.07, 0.02],
                        "_ctx_opt_skew_put_iv": [0.30, None, 0.25], "_ctx_opt_skew_call_iv": [0.25, 0.2, 0.23]})
    f = ce.signals_frame(log).set_index("Ticker")
    assert f.loc["A", "options_put_skew"] == pytest.approx(0.05)
    assert np.isnan(f.loc["B", "options_put_skew"]) and f.loc["C", "options_put_skew"] == pytest.approx(0.02)
    old = log.drop(columns=["_ctx_opt_skew_put_iv", "_ctx_opt_skew_call_iv"])
    assert ce.signals_frame(old)["options_put_skew"].isna().all(), "a log from before the legs is the old measure"
    assert "Xing, Zhang & Zhao" in ce.SIGNALS["options_put_skew"][0]


def test_the_harness_evaluates_every_option_reading_the_panel_shows():
    shown = {"om": "options_expected_move", "oi": "options_atm_iv", "osk": "options_put_skew"}
    log = _log(**{"Ticker": ["A"], "_ctx_last_close": [1.0], "_ctx_opt_status": ["ok"],
                  **{gd.CTX_FIELDS[k]: [0.1] for k in shown},
                  "_ctx_opt_skew_put_iv": [0.3], "_ctx_opt_skew_call_iv": [0.2]})
    f = ce.signals_frame(log)
    for key, signal in shown.items():
        assert signal in ce.SIGNALS and f[signal].iloc[0] == pytest.approx(0.1), signal


def test_every_new_claim_is_registered_and_resolves():
    for cid in ("context.price_trend", "context.momentum_skip", "context.macro_momentum_rule",
                "context.macro_series", "context.logged_every_run"):
        c = claims.claim(cid)
        assert callable(c.resolve()), cid
        assert all("test_context_layer_claims" in x or "test_" in x for x in c.checked_by)
