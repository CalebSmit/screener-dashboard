"""The Track record panel's sentences, against the code that makes them true.

Claims ``context.track_record``, ``context.track_luck`` and ``context.track_turnover``.
Research: ``research/2026-10-19-track-record.md``.

**Why this module exists.** Queue item 4 of ``plan/context-layer.md`` checks every sentence a
context panel prints against its source. For the track record the "source" is mostly the
arithmetic in ``track_record.py``, and reading it against the panel on 2026-10-19 found the panel
saying more than the arithmetic does:

* "measured forward from each run as it was published, with nothing backfilled" - the February
  snapshots were written on 2026-03-09 from those runs' saved output by
  ``improvement_engine.backfill_from_existing_runs``, and the first one predates the public page;
* "entry at the close of the first trading day after each run" - the 2026-08-10 rebalance run was
  made after that day's close (its own recorded prices are that close), and entered at it;
* "monthly holding periods" - they ran from 11 to 105 days;
* "the like-for-like benchmark" - RSP's price is after its fund costs; the top 25 bears none;
* "Top fifth minus bottom fifth +7.7%" - a difference of returns, in percentage points;
* "N changes" to the methodology - a count of changelog *dates*, several of which changed no score;
* "for a company that was acquired it usually understates it, because takeovers come at a
  premium" - both picks it was printed beside were priced flat by the screener's own records;
* the panel said "N months" with ``days / 30`` and the tile with ``days / 30.4``.

The lesson the layer already carries (item 2): **a claim whose check tests the plumbing instead of
the assertion is an unchecked claim.** ``test_entry_is_never_before_the_run`` passed throughout,
because it compared dates, and a date cannot tell a 02:00 run from a 19:00 one. So the tests here
build each figure from hand arithmetic, and the entry test feeds a run that saw its own close.
"""

from __future__ import annotations

import math
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import claims  # noqa: E402
import generate_dashboard as gd  # noqa: E402
import stock_summary  # noqa: E402
import track_record as tr  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"


class _Snap:
    def __init__(self, d, ranks):
        self.date, self.ranks = d, ranks


def _prices(n_names=10, start="2026-01-02", end="2026-03-31"):
    idx = pd.bdate_range(start, end)
    n = len(idx)
    rng = np.random.default_rng(7)
    data = {f"T{i}": 100 * np.cumprod(1 + rng.normal(0.0005 + 0.0002 * i, 0.01, n)) for i in range(n_names)}
    data["RSP"] = 100 * 1.0005 ** np.arange(n)
    data["SPY"] = 100 * 1.0007 ** np.arange(n)
    return pd.DataFrame(data, index=idx)


def _render_js() -> str:
    js = gd._js_context()
    return js[js.index("function renderTrack()"):js.index("// ---- rankings filter")]


# --------------------------------------------------------------------------- entry timing
def _snapshot(tmp_path, day: str, prices_for: pd.Series):
    pd.DataFrame({"Ticker": list(prices_for.index), "price_at_scoring": list(prices_for.values)}
                 ).to_parquet(tmp_path / f"{day}_abc.parquet")


def test_a_run_that_saw_its_own_close_enters_at_the_next_close(tmp_path):
    """The 2026-08-10 case: the run's recorded prices are its own date's close."""
    prices = _prices(n_names=80)
    names = [c for c in prices.columns if c.startswith("T")]
    _snapshot(tmp_path, "2026-01-07", prices.loc["2026-01-07", names])
    assert tr.runs_after_the_close(["2026-01-07"], prices, snap_dir=tmp_path) == ["2026-01-07"]
    ranks = {t: i + 1 for i, t in enumerate(names)}
    late = tr.build([_Snap("2026-01-07", ranks)], prices, date(2026, 3, 31), changelog=[], after_close=["2026-01-07"])
    assert late["periods"][0]["entry"] == "2026-01-08"
    assert late["after_close"] == ["2026-01-07"]
    on_time = tr.build([_Snap("2026-01-07", ranks)], prices, date(2026, 3, 31), changelog=[])
    assert on_time["periods"][0]["entry"] == "2026-01-07"


def test_a_run_made_before_the_close_enters_at_that_close(tmp_path):
    """The 02:00 data loop: the run's recorded prices are the previous trading day's close."""
    prices = _prices(n_names=80)
    names = [c for c in prices.columns if c.startswith("T")]
    _snapshot(tmp_path, "2026-01-07", prices.loc["2026-01-06", names])
    assert tr.runs_after_the_close(["2026-01-07"], prices, snap_dir=tmp_path) == []


def test_the_previous_list_is_held_until_the_late_runs_entry(tmp_path):
    prices = _prices(n_names=80)
    names = [c for c in prices.columns if c.startswith("T")]
    ranks = {t: i + 1 for i, t in enumerate(names)}
    out = tr.build([_Snap("2026-01-05", ranks), _Snap("2026-02-02", ranks)], prices, date(2026, 3, 31),
                   changelog=[], after_close=["2026-02-02"])
    assert out["periods"][0]["exit"] == "2026-02-02"       # held through the late run's own date
    assert out["periods"][1]["entry"] == "2026-02-03"


def test_a_snapshot_with_no_recorded_prices_is_named_as_untimed(tmp_path):
    pd.DataFrame({"Ticker": ["T1"], "Rank": [1]}).to_parquet(tmp_path / "2026-01-07_x.parquet")
    _snapshot(tmp_path, "2026-01-08", pd.Series({"T1": 1.0}))
    assert tr.untimed_runs(["2026-01-07", "2026-01-08"], snap_dir=tmp_path) == ["2026-01-07"]


# --------------------------------------------------------------------------- turnover and cost
def test_turnover_is_the_share_of_the_basket_sold():
    # drifted A 60% / B 40% -> equal B, C: sell all of A (60%), buy 10% more B and 50% C
    assert tr.one_way_turnover({"A": 0.6, "B": 0.4}, ["B", "C"]) == pytest.approx(0.6)
    assert tr.one_way_turnover(None, ["A"]) is None
    assert tr.one_way_turnover({"A": 0.5, "B": 0.5}, ["A", "B"]) == pytest.approx(0.0)


def test_turnover_in_the_record_counts_the_drift_by_hand(monkeypatch):
    monkeypatch.setattr(tr, "TOP_N", 2)
    prices = _prices()
    ranks1 = {f"T{i}": i + 1 for i in range(10)}                       # T0, T1 on top
    ranks2 = {**ranks1, "T0": 3, "T2": 1}                               # T2 replaces T0
    out = tr.build([_Snap("2026-01-05", ranks1), _Snap("2026-02-02", ranks2)], prices, date(2026, 3, 31), changelog=[])
    entry, exit_ = pd.Timestamp(out["periods"][0]["entry"]), pd.Timestamp(out["periods"][0]["exit"])
    g0 = prices.loc[exit_, "T0"] / prices.loc[entry, "T0"]
    g1 = prices.loc[exit_, "T1"] / prices.loc[entry, "T1"]
    w0, w1 = g0 / (g0 + g1), g1 / (g0 + g1)
    expected = 0.5 * (abs(0.5 - w1) + w0 + 0.5)                         # T1 trimmed or topped, T0 out, T2 in
    assert out["periods"][1]["turnover"] == pytest.approx(expected, abs=1e-4)
    assert out["periods"][0]["turnover"] is None                        # the first purchase is not turnover
    assert out["turnover"]["rebalances"] == 1


def test_the_break_even_cost_erases_the_lead_exactly():
    tos = [0.5, 0.3, 0.2]
    c = tr.break_even_cost(0.12, 0.06, tos)
    net = 1.12 * np.prod([1 - 2 * c * t for t in tos])
    assert net == pytest.approx(1.06, abs=1e-9)
    assert tr.break_even_cost(0.05, 0.06, tos) is None                 # no lead, nothing to erase
    assert tr.break_even_cost(0.12, 0.06, []) is None


# --------------------------------------------------------------------------- what a short record shows
def test_the_coin_toss_chance_is_the_binomial_tail():
    assert tr.coin_toss_chance(5, 7) == pytest.approx(29 / 128)
    assert tr.coin_toss_chance(0, 4) == pytest.approx(1.0)
    assert tr.coin_toss_chance(0, 0) is None


def test_the_excess_t_is_the_daily_mean_over_its_standard_error():
    idx = pd.bdate_range("2026-01-02", periods=120)
    rng = np.random.default_rng(3)
    rb = rng.normal(0.0004, 0.01, 120)
    ra = rb + rng.normal(0.0003, 0.005, 120)
    a = pd.Series(100 * np.cumprod(1 + ra), index=idx)
    b = pd.Series(100 * np.cumprod(1 + rb), index=idx)
    ex = (a.pct_change() - b.pct_change()).dropna()
    t = ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))
    got = tr.excess_record(a, b)
    assert got["days"] == len(ex) and got["t"] == pytest.approx(t, abs=0.01)
    assert got["years_to_two"] == pytest.approx((2 / t) ** 2 * len(ex) / 252, abs=0.1)


def test_the_record_publishes_no_annualised_figure():
    """GIPS 2020 for Firms 2.A.12: returns for periods of less than one year must not be
    annualised. Nothing in the record carries an annual rate, and the panel prints none."""
    out = tr.build([_Snap("2026-01-05", {f"T{i}": i + 1 for i in range(10)})], _prices(), date(2026, 3, 31), changelog=[])
    assert not any("annual" in k.lower() for k in _keys(out))
    js = _render_js()
    assert not re.search(r"(% a year|per year|annualised (return|lead)|annualized)", js)
    assert "No return here is annualised" in js


def _keys(d):
    if isinstance(d, dict):
        for k, v in d.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(d, list):
        for v in d:
            yield from _keys(v)


# --------------------------------------------------------------------------- the panel's words
def test_the_panel_does_not_claim_a_record_with_nothing_backfilled():
    js = _render_js()
    assert "nothing backfilled" not in js and "as it was published" not in js
    assert "the list each run recorded on its date" in js


def test_the_panel_says_no_money_was_invested():
    """GIPS 2020 4.C.48: theoretical performance must be disclosed as not based on actual assets."""
    js = _render_js()
    assert "No money was invested" in js and "model record" in js and "not a portfolio to follow" in js


def test_the_benchmark_is_described_with_its_costs_and_not_as_like_for_like():
    js = _render_js()
    assert "like-for-like" not in js
    assert "after the fund&rsquo;s own costs" in js and "before trading costs" in js


def test_the_spread_is_printed_in_percentage_points():
    js = _render_js()
    assert "cPp(T.spread)" in js and "cPct(T.spread)" not in js
    assert "does the ranking separate stocks" not in js


def test_the_periods_are_not_called_monthly_and_their_count_is_set_against_chance():
    js = _render_js()
    assert "monthly holding periods" not in js
    assert "a coin toss does at least this well" in js and "beat_rsp_chance" in js


def test_the_changelog_ticks_are_called_entries_not_changes():
    js = gd._js_context()
    assert "Methodology change (hover)" not in js and "Methodology changelog entry (hover)" in js
    assert "not every entry changed how stocks are scored" in _render_js()


def test_left_out_picks_are_sized_and_no_takeover_premium_is_asserted():
    js = _render_js()
    assert "takeovers come at a premium" not in js
    assert "picks in the" in js


def test_entry_sentence_names_the_late_and_the_untimed_runs():
    js = _render_js()
    assert "first trading day on or after its run&rsquo;s date" in js
    assert "T.after_close" in js and "T.untimed" in js


def test_the_panel_and_the_tile_count_months_the_same_way():
    js = gd._js_context()
    assert "function trMonths(T)" in js
    assert "const mo = trMonths(T);" in js and "trMonths(T) + ' months" in js
    assert "/ 30.4)" not in js and "T.days / 30)" not in js


def test_the_track_panel_carries_no_advice_language():
    js = _render_js()
    strings = " ".join(re.findall(r"'([^']*)'", js))
    assert not stock_summary.advice_terms_in(strings), stock_summary.advice_terms_in(strings)


@pytest.mark.parametrize("sentence", [
    "How the ranking has actually done since Feb 20, measured forward from each run as it was "
    "published, with nothing backfilled.",
    "Leaving them out can move a basket&rsquo;s return either way; for a company that was acquired "
    "it usually understates it, because takeovers come at a premium.",
])
def test_the_sentences_corrected_on_2026_10_19_stay_forbidden(sentence):
    assert claims.forbidden_hits(sentence)


# --------------------------------------------------------------------------- rendered
needs_browser = pytest.mark.skipif(sync_api is None or not PAGE.exists() or not PAYLOAD.exists(),
                                   reason="playwright or the generated dashboard is not present")


@needs_browser
def test_the_panel_renders_every_corrected_sentence_from_a_record():
    prices = _prices(n_names=80)
    names = [c for c in prices.columns if c.startswith("T")]
    ranks = {t: i + 1 for i, t in enumerate(names)}
    T = tr.build([_Snap("2026-01-05", ranks), _Snap("2026-02-02", {**ranks, names[0]: 80, names[79]: 1})],
                 prices, date(2026, 3, 31), changelog=[{"date": "2026-02-10", "title": "x"}],
                 after_close=["2026-02-02"], untimed=["2026-01-05"])
    T["unpriced"] = {"top": [], "q1": ["ZZZ"], "q5": []}
    import json
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        page = b.new_page(viewport={"width": 375, "height": 800})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(PAGE.as_uri(), wait_until="load", timeout=60000)
        page.wait_for_function("typeof renderTrack === 'function' && typeof D !== 'undefined'", timeout=30000)
        page.evaluate("t => { D.track = t; TR_DONE = true; renderTrack(); }", json.loads(json.dumps(T)))
        text = page.evaluate("document.getElementById('track-body').textContent")
        b.close()
    assert not errors, errors
    assert "No money was invested" in text
    assert "a coin toss does at least this well" in text
    assert "Traded at each rebalance" in text
    assert "standard errors from zero" in text
    assert "Feb 2" in text and "made after that close" in text
    assert "Jan 5 recorded no prices" in text
    assert "1 of the" in text and "top-fifth baskets" in text
    assert "pp" in text and "No return here is annualised" in text
    assert not stock_summary.advice_terms_in(text)
