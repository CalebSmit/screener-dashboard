"""No Yahoo insider request for a stock whose rows the SEC already supplies - same output.

research/2026-10-26-data-run-timing.md: at 02:00 option quotes are not served, so each request
in the context pass fetched only Yahoo's insider rows (paced at 0.35 s, ~178 s for 503 stocks),
and the SEC Form 4 step that ran next overwrote ``_ctx_insider`` for every one of them (503 of
503 on 2026-10-26). The SEC step now runs first and the pass skips the stocks it covers; the
records are set in the same order as before, so each record ends with the same fields in the
same order. These tests pin the skip, its limits (never with live quotes), the ordering in
``run_screener``, and the identity of the result.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

import context_fetch as cf

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def offline(monkeypatch):
    calls = []

    def fake_context_for(t, price, earnings_ts, today, with_options=True):
        calls.append((t, with_options))
        out = {"_ctx_insider": json.dumps([{"who": f"yahoo-{t}"}])}
        if with_options:
            out["_ctx_opt_status"] = "ok"
        return out

    import options_cache
    monkeypatch.setattr(cf, "_context_for", fake_context_for)
    monkeypatch.setattr(cf, "PACE_SECONDS", 0)
    monkeypatch.setattr(options_cache, "load", lambda root=None: {"quote_date": "2026-10-23"})
    monkeypatch.setattr(options_cache, "reading_for",
                        lambda cache, t, today: {"_ctx_opt_status": "cached", "_ctx_opt_em": 0.05}
                        if t != "C" else None)
    return calls


def _raw():
    return [{"Ticker": "A", "price_latest": 10.0}, {"Ticker": "B", "_error": "x"},
            {"Ticker": "C", "price_latest": 5.0}, {"Ticker": "D", "price_latest": 7.0}]


def test_quotes_closed_skips_only_the_covered_stocks(offline):
    raw = _raw()
    st = cf.enrich(raw, budget_seconds=60, log=lambda *a: None, options_live=False,
                   insider_from_sec=frozenset({"A", "C"}))
    assert [t for t, _ in offline] == ["D"]
    assert st["insider_from_sec"] == 2 and st["of"] == 3 and st["done"] == 1
    # the cached option reading still reaches every stock, covered or not
    assert raw[0]["_ctx_opt_status"] == "cached" and raw[2]["_ctx_opt_status"] == "quotes-closed"


def test_live_quotes_skip_nothing(offline):
    """With live quotes the request also fetches the option chain - it is never skipped."""
    st = cf.enrich(_raw(), budget_seconds=60, log=lambda *a: None, options_live=True,
                   insider_from_sec=frozenset({"A", "C", "D"}))
    assert sorted(t for t, _ in offline) == ["A", "C", "D"]
    assert st["insider_from_sec"] == 0


def test_no_sec_rows_means_the_old_pass(offline):
    st = cf.enrich(_raw(), budget_seconds=60, log=lambda *a: None, options_live=False)
    assert sorted(t for t, _ in offline) == ["A", "C", "D"] and st["insider_from_sec"] == 0


def _then_sec(raw, sec):
    """run_screener's assignment, unchanged: after the pass, the SEC rows replace Yahoo's."""
    for r in raw:
        if r.get("Ticker") in sec:
            r["_ctx_insider"] = sec[r["Ticker"]]
            r["_ctx_insider_src"] = "sec"
    return raw


@pytest.mark.parametrize("covered", [set(), {"A"}, {"A", "C", "D"}, {"A", "B", "C", "D"}])
def test_records_are_identical_with_and_without_the_skip(offline, covered):
    sec = {t: json.dumps([{"who": f"sec-{t}"}]) for t in covered}
    old = _then_sec(_after(cf.enrich, _raw(), frozenset()), sec)
    new = _then_sec(_after(cf.enrich, _raw(), frozenset(covered)), sec)
    assert old == new
    assert [list(r) for r in old] == [list(r) for r in new]       # same keys, same order


def _after(enrich, raw, skip):
    enrich(raw, budget_seconds=60, log=lambda *a: None, options_live=False, insider_from_sec=skip)
    return copy.deepcopy(raw)


def test_run_screener_fetches_sec_first_and_assigns_after_the_pass():
    src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    refresh = src.index("_cache = _ia.refresh(")
    enrich = src.index("stats[\"context_pass\"] = context_fetch.enrich(")
    assign = src.index('_r["_ctx_insider_src"] = "sec"')
    assert refresh < enrich < assign
    call = src[enrich:src.index(")", src.index("insider_from_sec", enrich)) + 1]
    assert "insider_from_sec=" in call
    # the SEC rows are taken for every record, as before - including one whose core fetch failed
    assert re.search(r"for _r in raw:\s+_rows = _ia\.sec_rows_for\(_cache, _r\.get\(\"Ticker\"\)", src)
