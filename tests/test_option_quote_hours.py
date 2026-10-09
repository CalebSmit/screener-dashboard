"""Option quotes exist only at some hours, and the page must say which one it used.

WHY THIS EXISTS - measured 2026-10-09, ``plan/context-layer.md`` queue item 2
(``research/measurements/2026-10-09-option-quote-availability.py``).

The context layer's options panel needs a quote. The source serves the chain around the clock but
overnight returns every strike with ``bid`` and ``ask`` of 0.00 and ``impliedVolatility`` of 0.000.
On the same code and the same day:

* 2026-10-08, 21:27 ET (owner-run): 394 of 503 stocks usable - **78.3%**
* 2026-10-09, 03:00 ET (the scheduled data loop): **0** of 503 - 484 ``stale-quotes``
* a live probe at 07:03 ET: 0 of 10, with ``lastPrice`` and ``openInterest`` intact

So the loop that publishes the site spent ~1,000 option requests a night and showed a number to
nobody, while the panel blamed quotes "missing or too wide at the time of the fetch" - which reads
as a transient glitch rather than the hour.

What is pinned here:

1. **Nothing loosens.** The fix is not to accept ``lastPrice`` in place of a quote. Battalio &
   Schultz (2006) show that apparent option mispricings largely vanish when quotes replace last
   trade prices, so a straddle built from last trades is exactly the wrong repair.
2. **Find out, do not guess the clock.** ``quotes_are_live`` probes real chains; a hard-coded
   window would go wrong the day the source changes.
3. **A reused reading says which session it came from**, and its days-to-expiry are counted from
   today, not from the quote date.
4. **A refresh at a dead hour leaves the cache alone** rather than overwriting good readings with
   500 empty ones.
5. **No context field reaches a score** - the standing rule for this whole layer.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

import pytest

import claims
import context_fetch
import options_cache

ROOT = Path(__file__).resolve().parent.parent
TODAY = datetime.date(2026, 10, 9)


def _reading(status="ok", expiry="2026-11-06", quote_date="2026-10-08"):
    return {"_ctx_opt_status": status, "_ctx_opt_move": 0.045, "_ctx_opt_iv": 0.27,
            "_ctx_opt_expiry": expiry, "_ctx_opt_days": 29,
            "_ctx_opt_quote_date": quote_date}


def _cache(**readings):
    return {"quote_date": "2026-10-08", "readings": dict(readings)}


# --------------------------------------------------------------------------- the probe
def test_quotes_count_as_live_as_soon_as_one_probe_is_usable():
    live, why = options_cache.quotes_are_live(lambda t: "ok" if t == "JPM" else "stale-quotes")
    assert live and "JPM:ok" in why


@pytest.mark.parametrize("status", ["stale-quotes", "no-chain", "no-atm", None])
def test_quotes_count_as_not_live_when_every_probe_comes_back_without_one(status):
    live, why = options_cache.quotes_are_live(lambda t: status)
    assert not live, why
    assert "no quotes being served" in why


def test_partial_is_usable_enough_to_fetch_the_rest():
    """'partial' means one of the straddle or the IV came back - the hour is alive."""
    live, _ = options_cache.quotes_are_live(lambda t: "partial")
    assert live


def test_an_all_errors_probe_tries_anyway_rather_than_skipping_the_night():
    """A network blip says nothing about the hour. Failing closed here would silently drop the
    option half of the pass on any flaky night."""
    def boom(_t):
        raise RuntimeError("connection reset")

    live, why = options_cache.quotes_are_live(boom)
    assert live and "inconclusive" in why


def test_the_probe_stops_at_the_limit_and_does_not_walk_the_universe():
    asked: list[str] = []

    def record(t):
        asked.append(t)
        return "stale-quotes"

    options_cache.quotes_are_live(record, tickers=[f"T{i}" for i in range(50)], limit=3)
    assert len(asked) == 3, asked


# --------------------------------------------------------------------------- reading back
def test_a_served_reading_carries_its_quote_date():
    got = options_cache.reading_for(_cache(AAA=_reading()), "AAA", TODAY)
    assert got["_ctx_opt_quote_date"] == "2026-10-08"
    assert got["_ctx_opt_move"] == 0.045


def test_days_to_expiry_are_recounted_from_today():
    """The panel prints '(N days)'. N shrinks every day a reading is reused; serving the stored
    29 would overstate the time left."""
    got = options_cache.reading_for(_cache(AAA=_reading()), "AAA", TODAY)
    assert got["_ctx_opt_days"] == (datetime.date(2026, 11, 6) - TODAY).days == 28


def test_a_reading_older_than_the_limit_is_refused():
    old = _reading(quote_date="2026-10-01")
    assert options_cache.reading_for(_cache(AAA=old), "AAA", TODAY) is None


def test_a_friday_close_still_serves_on_monday():
    """Three days is chosen to carry a Friday close over a weekend; a shorter limit would make
    every Monday empty."""
    friday, monday = "2026-10-09", datetime.date(2026, 10, 12)
    assert datetime.date(2026, 10, 9).weekday() == 4 and monday.weekday() == 0
    got = options_cache.reading_for(_cache(AAA=_reading(quote_date=friday)), "AAA", monday)
    assert got is not None and got["_ctx_opt_quote_date"] == friday


def test_a_reading_whose_expiry_has_passed_is_refused():
    assert options_cache.reading_for(_cache(AAA=_reading(expiry="2026-10-02")),
                                     "AAA", TODAY) is None


def test_a_future_quote_date_is_refused():
    assert options_cache.reading_for(_cache(AAA=_reading(quote_date="2026-10-20")),
                                     "AAA", TODAY) is None


def test_a_missing_or_malformed_entry_is_refused_not_raised():
    for bad in ({"readings": {}}, {"readings": {"AAA": "nonsense"}}, {},
                _cache(AAA={"_ctx_opt_status": "ok"})):
        assert options_cache.reading_for(bad, "AAA", TODAY) is None


def test_an_unreadable_cache_file_is_ignored_not_fatal(tmp_path):
    p = options_cache.cache_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ this is not json", encoding="utf-8")
    with pytest.warns(UserWarning, match="unreadable"):
        assert options_cache.load(tmp_path) == {"readings": {}}


def test_save_then_load_round_trips(tmp_path):
    options_cache.save({"AAA": _reading()}, "2026-10-08", root=tmp_path)
    c = options_cache.load(tmp_path)
    assert c["quote_date"] == "2026-10-08" and "AAA" in c["readings"]
    assert "written" in c


# --------------------------------------------------------------------------- the refresh pass
def test_a_refresh_at_a_dead_hour_writes_nothing(tmp_path):
    """The failure this prevents: a 2 AM refresh replacing 400 good readings with 500 empties."""
    options_cache.save({"AAA": _reading()}, "2026-10-08", root=tmp_path)
    before = options_cache.cache_path(tmp_path).read_bytes()
    stats = options_cache.refresh([{"Ticker": "AAA", "price_latest": 10.0}], root=tmp_path,
                                  log=lambda *_a: None,
                                  fetch=lambda *a, **k: {"_ctx_opt_status": "stale-quotes"})
    assert stats["live"] is False and stats["kept"] == 0
    assert options_cache.cache_path(tmp_path).read_bytes() == before


def test_a_refresh_keeps_only_usable_readings(tmp_path):
    exp = (TODAY + datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    def fetch(ticker, *_a, **_k):
        status = {"AAA": "ok", "BBB": "partial", "CCC": "stale-quotes", "DDD": "no-chain"}[ticker]
        return {"_ctx_opt_status": status, "_ctx_opt_expiry": exp, "_ctx_opt_move": 0.04}

    records = [{"Ticker": t, "price_latest": 10.0} for t in ("AAA", "BBB", "CCC", "DDD")]
    stats = options_cache.refresh(records, root=tmp_path, log=lambda *_a: None, fetch=fetch,
                                  probe=lambda _t: "ok")
    assert stats["kept"] == 2
    kept = options_cache.load(tmp_path)["readings"]
    assert set(kept) == {"AAA", "BBB"}
    assert all(r["_ctx_opt_quote_date"] for r in kept.values())


def test_a_refresh_stores_only_option_fields(tmp_path):
    """The cache is for quotes. Letting a trend or insider field in would mean the 02:00 run read
    a stale copy of data it fetches fresh every night."""
    exp = (TODAY + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    options_cache.refresh([{"Ticker": "AAA", "price_latest": 10.0}], root=tmp_path,
                          log=lambda *_a: None, probe=lambda _t: "ok",
                          fetch=lambda *a, **k: {"_ctx_opt_status": "ok", "_ctx_opt_expiry": exp,
                                                 "_ctx_last_close": 10.0, "_ctx_insider": "[]"})
    stored = options_cache.load(tmp_path)["readings"]["AAA"]
    assert all(k.startswith("_ctx_opt") for k in stored), stored


def test_a_refresh_merges_rather_than_dropping_stocks_it_did_not_reach(tmp_path):
    """A budget-stopped refresh must not delete yesterday's readings for the stocks it missed."""
    exp = (TODAY + datetime.timedelta(days=40)).strftime("%Y-%m-%d")
    options_cache.save({"OLD": _reading(expiry=exp,
                                        quote_date=TODAY.strftime("%Y-%m-%d"))},
                       TODAY.strftime("%Y-%m-%d"), root=tmp_path)
    options_cache.refresh([{"Ticker": "NEW", "price_latest": 10.0}], root=tmp_path,
                          log=lambda *_a: None, probe=lambda _t: "ok",
                          fetch=lambda *a, **k: {"_ctx_opt_status": "ok", "_ctx_opt_expiry": exp})
    assert set(options_cache.load(tmp_path)["readings"]) == {"OLD", "NEW"}


def test_a_refresh_drops_entries_that_can_no_longer_be_served(tmp_path):
    """Otherwise the file grows for every ticker that ever had a reading."""
    exp = (TODAY + datetime.timedelta(days=40)).strftime("%Y-%m-%d")
    options_cache.save({"DEAD": _reading(expiry="2020-01-17", quote_date="2020-01-02")},
                       "2020-01-02", root=tmp_path)
    options_cache.refresh([{"Ticker": "NEW", "price_latest": 10.0}], root=tmp_path,
                          log=lambda *_a: None, probe=lambda _t: "ok",
                          fetch=lambda *a, **k: {"_ctx_opt_status": "ok", "_ctx_opt_expiry": exp})
    assert set(options_cache.load(tmp_path)["readings"]) == {"NEW"}


# --------------------------------------------------------------------------- the 02:00 pass
def test_the_context_pass_skips_the_option_fetch_when_quotes_are_closed(tmp_path, monkeypatch):
    """The whole point: ~1,000 requests a night that produced nothing for anybody."""
    calls: list[str] = []
    monkeypatch.setattr(context_fetch, "_options_for",
                        lambda *a, **k: calls.append(a[0]) or {})
    monkeypatch.setattr(context_fetch, "_context_for",
                        lambda t, p, e, d, with_options=True:
                        (calls.append(t) if with_options else None) or {"_ctx_insider": "[]"})
    raw = [{"Ticker": t, "price_latest": 10.0} for t in ("AAA", "BBB")]
    stats = context_fetch.enrich(raw, log=lambda *_a: None, root=tmp_path, options_live=False)
    assert calls == [], f"fetched option chains at a dead hour: {calls}"
    assert stats["options_live"] is False
    assert [r["_ctx_opt_status"] for r in raw] == [options_cache.STATUS_CLOSED] * 2


def test_the_context_pass_serves_the_cache_when_quotes_are_closed(tmp_path, monkeypatch):
    exp = (datetime.datetime.now(datetime.timezone.utc).date()
           + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    qd = datetime.datetime.now(datetime.timezone.utc).date().strftime("%Y-%m-%d")
    options_cache.save({"AAA": _reading(expiry=exp, quote_date=qd)}, qd, root=tmp_path)
    monkeypatch.setattr(context_fetch, "_context_for",
                        lambda *a, **k: {"_ctx_insider": "[]"})
    raw = [{"Ticker": t, "price_latest": 10.0} for t in ("AAA", "BBB")]
    context_fetch.enrich(raw, log=lambda *_a: None, root=tmp_path, options_live=False)
    assert raw[0]["_ctx_opt_status"] == "ok" and raw[0]["_ctx_opt_quote_date"] == qd
    assert raw[1]["_ctx_opt_status"] == options_cache.STATUS_CLOSED


def test_the_insider_half_still_runs_when_quotes_are_closed(tmp_path, monkeypatch):
    """Form 4 data does not care what hour it is; only the option half has the problem."""
    monkeypatch.setattr(context_fetch, "_context_for",
                        lambda *a, **k: {"_ctx_insider": '[{"code":"P"}]'})
    raw = [{"Ticker": "AAA", "price_latest": 10.0}]
    stats = context_fetch.enrich(raw, log=lambda *_a: None, root=tmp_path, options_live=False)
    assert json.loads(raw[0]["_ctx_insider"]) == [{"code": "P"}]
    assert stats["done"] == 1


def test_a_cached_reading_survives_the_insider_update(tmp_path, monkeypatch):
    """Both halves write into the same record. The insider pass returning no option fields must
    not blank the ones the cache just supplied."""
    exp = (datetime.datetime.now(datetime.timezone.utc).date()
           + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    qd = datetime.datetime.now(datetime.timezone.utc).date().strftime("%Y-%m-%d")
    options_cache.save({"AAA": _reading(expiry=exp, quote_date=qd)}, qd, root=tmp_path)
    monkeypatch.setattr(context_fetch, "_context_for",
                        lambda *a, **k: {"_ctx_insider": "[]", "_ctx_opt_status": "stale-quotes"})
    raw = [{"Ticker": "AAA", "price_latest": 10.0}]
    context_fetch.enrich(raw, log=lambda *_a: None, root=tmp_path, options_live=False)
    assert raw[0]["_ctx_opt_status"] == "ok"


def test_the_context_pass_still_fetches_options_when_quotes_are_live(tmp_path, monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(context_fetch, "_context_for",
                        lambda t, p, e, d, with_options=True:
                        seen.append(with_options) or {"_ctx_insider": "[]"})
    context_fetch.enrich([{"Ticker": "AAA", "price_latest": 10.0}], log=lambda *_a: None,
                         root=tmp_path, options_live=True)
    assert seen == [True]


# --------------------------------------------------------------------------- honesty on the page
def _generator_source() -> str:
    return (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")


def test_the_page_names_the_quote_date_when_it_is_not_todays():
    src = _generator_source()
    assert "oqd" in src, "the quote date must reach the payload"
    assert '"oqd": "_ctx_opt_quote_date"' in src
    # the sentence that uses it
    block = src[src.index("Expected move = at-the-money call + put"):]
    block = block[:block.index("</p>")]
    assert "c.oqd" in block and "Quotes are the close of" in block, block[:400]


def test_the_page_explains_the_closed_hour_rather_than_blaming_the_quotes():
    src = _generator_source()
    assert f"c.os === '{options_cache.STATUS_CLOSED}'" in src
    i = src.index(f"c.os === '{options_cache.STATUS_CLOSED}'")
    sentence = src[i:i + 700]
    assert "2 AM" in sentence and "bid and ask at zero" in sentence, sentence[:300]


def test_the_closed_status_says_nothing_that_reads_as_advice():
    from stock_summary import advice_terms_in

    src = _generator_source()
    i = src.index(f"c.os === '{options_cache.STATUS_CLOSED}'")
    sentence = re.sub(r"<[^>]+>", " ", src[i:i + 700])
    assert not advice_terms_in(sentence), advice_terms_in(sentence)


def test_the_quote_date_claim_is_registered():
    c = {x.id: x for x in claims.CLAIMS}["context.expected_move"]
    assert "session's date" in c.asserts or "session&rsquo;s" in c.asserts
    assert any("test_option_quote_hours" in r for r in c.checked_by)


# --------------------------------------------------------------------------- the standing rule
def test_no_scoring_module_imports_the_option_cache():
    """CLAUDE.md settled row 'ctx': context is shown beside the score, never in it."""
    for name in ("factor_engine.py", "run_screener.py", "portfolio_constructor.py",
                 "improvement_engine.py", "backtest.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        for line in src.splitlines():
            s = line.strip()
            if s.startswith(("import options_cache", "from options_cache")):
                # run_screener orchestrates the pipeline; it may wire the pass, not score from it
                assert name == "run_screener.py", f"{name} imports options_cache"


def test_the_cache_is_not_tracked_by_git():
    """It is re-downloadable, changes every day, and would otherwise be committed by the data
    loop - the same reasoning as data/market/, data/track/, data/insider/."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/options/" in ignore


def test_last_trade_prices_are_not_used_in_place_of_quotes():
    """The tempting repair, ruled out: Battalio & Schultz (2006) show apparent option
    mispricings largely vanish when quotes replace last trade prices, so a straddle built from
    lastPrice is the wrong fix. See the module docstring."""
    src = (ROOT / "context_signals.py").read_text(encoding="utf-8")
    block = src[src.index("def _mid("):src.index("def choose_expiry(")]
    assert "lastPrice" not in block, "the straddle must come from bid/ask, not the last trade"
