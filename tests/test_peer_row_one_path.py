"""One company, one set of figures: the peer table's rows all go through ``buildPeerRow``.

2026-11-08. The peer comparison table prints nine figures a row - market cap, P/E, revenue
growth, net margin, ROIC (or ROE for a bank), debt/equity, dividend yield, free-cash-flow
yield - for the stock whose sheet is open and for each of its sector peers.

Until today it built those rows **two ways**. The open stock's row came from a precomputed
``stock_detail[t].self_metrics`` block written in Python at build time; every peer's row came
from ``buildPeerRow`` in JS over ``stock_detail[t].raw`` and ``stock_detail[t].financials``.
Every stock sits on both sides of that split - it is the subject of its own sheet and a peer in
its sector-mates' - and the two expressions were not textually the same: P/E was
``round(1.0 / earnings_yield, 1)`` on the unrounded yield in Python (round-half-to-even) against
``Math.round(10.0 / ey) / 10`` on the published 4dp yield in JS (half away from zero).

**Measured before the change** (``research/measurements/2026-11-08-peer-row-two-paths.py``):
**0 of 4,491 cells disagreed** on the 2026-11-06 payload, so no published figure moved when the
block was removed. The reason to remove it anyway is CLAUDE.md row 0.8 - the page shows the
engine's numbers and never a second derivation of them - and one path cannot disagree with
itself. It also took **16.7 KB off first paint** (1.240 -> 1.223 MB gzipped) for figures the
payload already carried in ``raw`` and ``financials``, which is the cheapest slice of
`plan/universe-expansion.md` phase 1 available: no fetch, no waiting state, nothing on demand.

The test that would have caught the duplication is
``test_a_company_prints_the_same_row_on_its_own_sheet_and_as_a_peer``: it reads one company's
row from its own sheet and the same company's row from another stock's sheet, and requires the
printed cells to be character-identical. No assertion about which stock - both are chosen by
``tests/examples.py`` (settled row "data->red").
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
from tests import examples  # noqa: E402

PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"
WAIT = 8000
CATEGORIES = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]

# The nine figures a row carries. ROIC and ROE share one printed column (the subject's
# `flags.is_bank` chooses), so the table prints eight figure columns plus the rank.
PEER_COLS = ("mcap", "pe_ratio", "rev_growth", "net_margin", "roic", "roe",
             "debt_equity", "div_yield", "fcf_yield")
# <td> positions in a row: 0 ticker, 1 rank, 2..9 the eight figure columns, 10 the remove button.
FIGURES = slice(2, 10)


def _payload() -> dict:
    txt = PAYLOAD.read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


# --------------------------------------------------------------- 1. the generator, not the file
def _frame() -> pd.DataFrame:
    df = pd.DataFrame({
        "Ticker": ["AAA", "BBB"],
        "Company": ["Alpha Inc", "Beta Corp"],
        "Sector": ["Information Technology", "Information Technology"],
        "Composite": [80.0, 60.0],
        "Rank": [1, 2],
        "Value_Trap_Flag": [False, False],
        "Growth_Trap_Flag": [False, False],
        "revenue_growth": [0.12, 0.03],
        "earnings_yield": [0.0803, 0.0500],
        "roic": [0.21, 0.08],
        "roe": [0.33, 0.11],
        "debt_equity": [0.45, 1.20],
        "fcf_yield": [0.06, 0.02],
        "_mcap": [3.0e12, 1.5e10],
        "_total_revenue": [4.0e11, 2.0e10],
        "_net_income": [1.0e11, 1.0e9],
        "dividend_yield": [0.0044, 0.0210],
    })
    for cat in CATEGORIES:
        df[cat + "_score"] = 50.0
    return df


@pytest.fixture(scope="module")
def built() -> dict:
    return json.loads(gd.prepare_dashboard_data({
        "df": _frame(), "meta": {"run_date": "2026-11-08"},
        "weights": {}, "sens_df": None, "corr_df": None, "cfg": {},
    }))


def test_the_generator_publishes_no_precomputed_self_row(built):
    """Asserted against the generator's output, not the committed payload (CLAUDE.md rule 10):
    a test reading a generated artifact passes on a hand-edit and says nothing about what the
    next run will publish."""
    for ticker, s in built["stock_detail"].items():
        assert "self_metrics" not in s, f"{ticker} still carries a precomputed self row"


def test_the_figures_the_row_needs_are_still_published(built):
    """Removing the block must not remove the data. Every one of the nine columns has to be
    reachable from ``raw`` or ``financials``, or the page would print em-dashes instead."""
    for ticker, s in built["stock_detail"].items():
        raw, fin = s.get("raw") or {}, s.get("financials") or {}
        for key in ("revenue_growth", "earnings_yield", "roic", "roe", "debt_equity", "fcf_yield"):
            assert key in raw, f"{ticker}: raw.{key} is missing, so the peer row cannot print it"
        for key in ("market_cap", "net_margin", "dividend_yield"):
            assert key in fin, f"{ticker}: financials.{key} is missing, so the peer row cannot print it"


def test_the_two_expressions_for_price_to_earnings_now_have_one_source(built):
    """The divergence the removal closes, stated as arithmetic rather than as a diff.

    ``AAA``'s earnings yield is 0.0803, where ``1 / ey`` is 12.4533...: both roundings give
    12.5, which is why nothing a reader sees moved. What matters is that the payload now
    publishes **one** input for the column - the 4dp ``raw.earnings_yield`` - so there is no
    second, unrounded copy for a different rounding rule to be applied to."""
    aaa = built["stock_detail"]["AAA"]
    assert aaa["raw"]["earnings_yield"] == 0.0803
    assert round(1.0 / 0.0803, 1) == 12.5
    assert "pe_ratio" not in aaa, "a published P/E would be a second source for the column"


# --------------------------------------------------------------- 2. the published payload
def test_the_page_contains_no_reader_of_a_precomputed_self_row():
    """The guard that makes the order of a reopening irrelevant.

    Measured (``research/measurements/2026-11-08-peer-row-mutations.py``): putting the JS read
    back on its own - ``const self = s.self_metrics || buildPeerRow(ticker)`` - **gets through**
    every other check here, and correctly so, because with no such block in the payload the
    fallback is all that ever runs. It is an equivalent mutant today and a loaded one tomorrow:
    a later session that reintroduces the block would find the page already preferring it. The
    pair together is caught (by ``test_the_generator_publishes_no_precomputed_self_row``); this
    makes either half alone enough."""
    page = gd.generate_html(methodology_html="", data_timestamp="x",
                            data_version="v", context_version="v", detail_version="v")
    # a property *read*, not the word - the code comments name the removed block on purpose,
    # so that the next reader finds the reasoning rather than rediscovering it
    reads = re.findall(r"(?:\.self_metrics\b|\[\s*['\"]self_metrics['\"]\s*\])", page)
    assert not reads, f"the generated page reads a precomputed self row again: {reads}"
    # and the nine columns are assembled in exactly one place
    assert page.count("function buildPeerRow(") == 1
    assert page.count("const selfRow = { ...self, isSelf: true }") == 1


@pytest.mark.skipif(not PAYLOAD.exists(), reason="no published payload")
def test_the_published_payload_carries_no_self_metrics():
    sd = _payload()["stock_detail"]
    carried = [t for t, s in sd.items() if "self_metrics" in s]
    assert not carried, carried[:10]


@pytest.mark.skipif(not PAYLOAD.exists(), reason="no published payload")
def test_the_empty_peer_state_is_a_guard_and_not_a_live_state():
    """"No peer figures were published for this stock in this run." must be unreachable on a good
    run - the panel's own premise, asserted rather than assumed. It fires when every one of the
    nine columns is null, which was true of 0 of 499 stocks when this was written (the thinnest
    carried six)."""
    sd = _payload()["stock_detail"]
    thin = {}
    for t, s in sd.items():
        raw, fin = s.get("raw") or {}, s.get("financials") or {}
        ey = raw.get("earnings_yield")
        have = [
            fin.get("market_cap"), (10.0 / ey) if (ey or 0) > 0 else None,
            raw.get("revenue_growth"), fin.get("net_margin"), raw.get("roic"), raw.get("roe"),
            raw.get("debt_equity"), fin.get("dividend_yield"), raw.get("fcf_yield"),
        ]
        n = sum(1 for v in have if v is not None)
        if n == 0:
            thin[t] = n
    assert not thin, f"these stocks would show the empty-state sentence: {thin}"


# --------------------------------------------------------------- 3. the page
try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def browser():
    if sync_api is None:
        pytest.skip("MISSING DATA: playwright is not installed")
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"MISSING DATA: chromium is unavailable ({type(e).__name__})")
        try:
            yield b
        finally:
            b.close()


def _open(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri(), wait_until="load", timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0",
                           timeout=30000)
    return ctx, page, errors


def _sheet(page, t):
    page.evaluate(f"openStockDetail({json.dumps(t)})")
    page.wait_for_function("document.getElementById('stock-modal').style.display === 'flex'",
                           timeout=WAIT)
    page.wait_for_selector("#modal-peers .peer-table", state="attached", timeout=WAIT)


def _row_cells(page, selector):
    return page.evaluate("""(sel) => { const r = document.querySelector(sel);
        return r ? [...r.querySelectorAll('td')].map(e => e.textContent.trim()) : null; }""", selector)


@pytest.mark.skipif(not PAGE.exists() or not PAYLOAD.exists(), reason="generated dashboard not present")
def test_the_subject_row_is_what_build_peer_row_returns(browser):
    """The unification itself, read off the live page: the row marked YOU is
    ``buildPeerRow(ticker)`` and nothing else. Any reintroduced second source would have to
    disagree with this to be worth having."""
    a = examples.stock("EXPE")
    ctx, page, errors = _open(browser)
    try:
        _sheet(page, a)
        same = page.evaluate("""(t) => { const r = buildPeerRow(t);
            return JSON.stringify(r) === JSON.stringify(buildPeerRow(t)) && r !== null; }""", a)
        assert same
        cells = _row_cells(page, "#modal-peers tr.peer-row-self")
        assert cells and cells[0].startswith(a), cells
        # the figures are the ones buildPeerRow produced, formatted
        n_dash = page.evaluate("""(t) => { const r = buildPeerRow(t);
            return ['mcap','pe_ratio','rev_growth','net_margin','debt_equity','div_yield','fcf_yield']
                .filter(k => r[k] === null || r[k] === undefined).length; }""", a)
        assert cells[FIGURES].count("\u2014") >= n_dash, (cells, n_dash)
        assert errors == []
    finally:
        ctx.close()


@pytest.mark.skipif(not PAGE.exists() or not PAYLOAD.exists(), reason="generated dashboard not present")
def test_a_company_prints_the_same_row_on_its_own_sheet_and_as_a_peer(browser):
    """The check the duplication could have failed: one company, two sheets, identical cells.

    ``a`` is read from its own sheet (the YOU row) and then from ``b``'s sheet, where it is added
    to the comparison the way a reader adds a name. Both are full non-bank sheets, so the shared
    ROIC/ROE column means the same thing on both - asserted, not assumed, because the column's
    label follows the *subject's* ``flags.is_bank``."""
    a = examples.stock("EXPE")
    b = examples.stock("HST", exclude=(a,))
    sd = _payload()["stock_detail"]
    assert a != b
    for t in (a, b):
        assert not ((sd[t].get("flags") or {}).get("is_bank")), f"{t} is bank-like; the profit column differs"

    ctx, page, errors = _open(browser)
    try:
        _sheet(page, a)
        own = _row_cells(page, "#modal-peers tr.peer-row-self")
        assert own, "no YOU row on the subject's own sheet"

        _sheet(page, b)
        page.evaluate("(t) => addPeer(t)", a)
        page.wait_for_function("""(t) => [...document.querySelectorAll('#modal-peers tr.peer-row td.peer-td-ticker')]
            .some(e => e.textContent.trim().startsWith(t))""", arg=a, timeout=WAIT)
        as_peer = page.evaluate("""(t) => { const r = [...document.querySelectorAll('#modal-peers tr.peer-row')]
            .find(x => x.querySelector('td.peer-td-ticker').textContent.trim().startsWith(t));
            return r ? [...r.querySelectorAll('td')].map(e => e.textContent.trim()) : null; }""", a)
        assert as_peer, f"{a} did not appear as a peer on {b}'s sheet"

        # not vacuous: eight figure columns, most of them carrying a real figure
        assert len(own[FIGURES]) == 8, own
        assert sum(1 for c in own[FIGURES] if c and c != "—") >= 6, own[FIGURES]
        assert own[1].startswith("#"), own[1]

        assert own[1] == as_peer[1], f"rank differs: {own[1]} vs {as_peer[1]}"
        assert own[FIGURES] == as_peer[FIGURES], (
            f"{a} prints different figures on its own sheet and on {b}'s:\n"
            f"  own:     {own[FIGURES]}\n  as peer: {as_peer[FIGURES]}")
        assert errors == []
    finally:
        ctx.close()
