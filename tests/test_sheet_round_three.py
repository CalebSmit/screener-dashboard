"""The stock sheet, round three (owner-run, 2026-10-27; ``plan/ui-elevation.md`` "Round three - the
sheet").

What this pins:

- **the scroll-spy** - a tab the reader chose stays lit until they scroll by hand (Price targets and
  Peers used to light Data, because the sheet ends before they reach its top), and scrolling freely
  lights every shown tab in order over the sheet's last screen;
- **Company snapshot has a tab** ("Financials"), in reading order;
- **the first open is warm** - once the page is idle the top stock's sheet is laid out unseen, one
  section at a time, and leaves nothing behind: the sheet closed, no history entry, no title, focus
  untouched, and a reader who opens a stock first ends it;
- **an unscored category says why** (claim ``drilldown.unscored_category``): the engine's rule, the
  published payload agreeing with it on every stock, and the display-only ``flags.px_rejected``;
- **empty sections say so**, and **sector peers** open their own sheet, show their rank and name
  their remove buttons.

Presentation only, except ``flags.px_rejected`` (display-only, never scored). Browser tests skip
loudly where Playwright or Chromium is missing, like ``test_dashboard_browser``; every wait is a
condition with an 8 s timeout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import generate_dashboard as gd

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"
WAIT = 8000
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]

pytestmark = pytest.mark.skipif(not PAGE.exists() or not PAYLOAD.exists(),
                                reason="generated dashboard not present")


def _payload() -> dict:
    txt = PAYLOAD.read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


@pytest.fixture(scope="module")
def cfg():
    import yaml
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# an unscored category, without a browser
# ---------------------------------------------------------------------------

def test_the_engine_leaves_a_category_unscored_only_when_no_weighted_metric_has_data(cfg):
    """The rule the sheet states, run through the engine itself: blank every weighted Momentum
    metric for one stock and Momentum goes NaN; leave one and it is scored; blank only a
    zero-weight one and nothing changes."""
    from factor_engine import compute_category_scores, compute_sector_percentiles, metric_weight_profiles
    from tests.test_scoring import _make_df

    weighted = [m for m, w in metric_weight_profiles(cfg, "momentum")["generic"].items() if w > 0]
    unweighted = [m for m, w in metric_weight_profiles(cfg, "momentum")["generic"].items() if not w > 0]
    assert weighted, "momentum has no weighted metric"

    def score(blank):
        df = _make_df(60)
        v = df.index[0]
        for m in blank:
            df.loc[v, m] = np.nan
        df = compute_category_scores(compute_sector_percentiles(df), cfg)
        return df.loc[v, "momentum_score"]

    assert pd.isna(score(weighted))
    assert pd.notna(score(weighted[1:]))
    if unweighted:
        assert pd.notna(score(unweighted))


def test_every_published_unscored_category_has_no_weighted_metric_with_data():
    """On the published payload, in both directions: a category is null exactly when no metric
    carrying weight in the table this stock was scored with (``wp``) has a percentile - so the
    reason the sheet prints is the engine's, for every stock."""
    D = _payload()
    profiles = D["weights"]["profiles"]
    unscored = 0
    for t, s in D["stock_detail"].items():
        for c in CATS:
            table = profiles[c].get((s.get("wp") or {}).get(c) or "generic") or profiles[c]["generic"]
            weighted = [m for m, w in table.items() if w and w > 0]
            has = [m for m in weighted if s["pct"].get(m) is not None]
            assert (s["cat_scores"].get(c) is None) == (not has), (t, c, has)
            unscored += s["cat_scores"].get(c) is None
    assert unscored >= 1, "no unscored category in this run - the browser test below has nothing to show"


def test_the_price_history_categories_are_built_only_from_the_price_history(cfg):
    """The sheet names a rejected price history on Momentum and Risk only, because every metric
    that carries weight in either - in every weight table - is withheld when the series is
    rejected (``compute_metrics`` on a rejected record), and the page's list says the same."""
    from factor_engine import PRICE_SERIES_DERIVED_FIELDS, compute_metrics, metric_weight_profiles
    raw = {"Ticker": "TEST", "Company": "Test Co", "Sector": "Consumer Staples",
           "currentPrice": 48.73, "fiftyTwoWeekHigh": 50.17,
           "_price_series_rejected": "series mixes pre- and post-split prices"}
    for f in PRICE_SERIES_DERIVED_FIELDS:
        raw[f] = np.nan
    row = compute_metrics([raw], pd.Series(dtype=float), risk_free_rate=0.045).iloc[0]
    for cat in ("momentum", "risk"):
        for table in metric_weight_profiles(cfg, cat).values():
            for m, w in table.items():
                if w and w > 0:
                    assert pd.isna(row[m]), (cat, m, row[m])
    js = gd._js_workings()
    assert "const PRICE_HISTORY_CATS = ['momentum', 'risk'];" in js


def test_the_rejection_reason_is_published_and_never_scored():
    """``flags.px_rejected`` carries the engine's own reason string, null otherwise, and is
    display-only: no scoring module reads it and it never appears in ``raw`` / ``pct``."""
    from tests.test_bot_payload_contract import _frame
    df = _frame()
    df["_price_series_rejected"] = ["series mixes pre- and post-split prices across a 2:1 split", None, np.nan]
    detail = json.loads(gd.prepare_dashboard_data({
        "df": df, "meta": {"run_date": "2026-10-27", "start_time": "2026-10-27T02:00:00"},
        "weights": {}, "sens_df": None, "corr_df": None, "cfg": {},
    }))["stock_detail"]
    assert detail["AAA"]["flags"]["px_rejected"].startswith("series mixes")
    assert detail["BRK-B"]["flags"]["px_rejected"] is None
    assert detail["CCC"]["flags"]["px_rejected"] is None
    for s in detail.values():
        assert "px_rejected" not in (s.get("raw") or {}) and "px_rejected" not in (s.get("pct") or {})
    for mod in ("factor_engine.py", "run_screener.py", "portfolio_constructor.py", "presets.py", "calc_trace.py"):
        assert "px_rejected" not in (ROOT / mod).read_text(encoding="utf-8"), mod


# ---------------------------------------------------------------------------
# markup
# ---------------------------------------------------------------------------

def test_company_snapshot_has_a_tab_in_reading_order():
    html = PAGE.read_text(encoding="utf-8")
    nav = html[html.index('class="modal-nav"'):]
    nav = nav[:nav.index("</nav>")]
    targets = re.findall(r"goToModal\('([a-z-]+)'\)", nav)
    assert targets[-4:] == ["section-price-targets", "section-snapshot", "section-peers", "section-provenance"]
    assert ">Financials</a>" in nav


# ---------------------------------------------------------------------------
# in a browser
# ---------------------------------------------------------------------------

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _open(browser, width=1280, height=900, touch=False, hash_=""):
    ctx = browser.new_context(viewport={"width": width, "height": height}, has_touch=touch, is_mobile=touch)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri() + hash_, wait_until="load", timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
    return ctx, page, errors


def _sheet(page, t):
    page.evaluate(f"openStockDetail({json.dumps(t)})")
    page.wait_for_function("document.getElementById('stock-modal').style.display === 'flex'", timeout=WAIT)


LIT = "(() => { const a = document.querySelector('#stock-modal .modal-nav a.active'); return a ? a.getAttribute('href').slice(1) : null; })()"


def test_a_chosen_tab_stays_lit_at_the_end_of_the_sheet(browser):
    """APA at 1280: Price targets and Peers cannot reach the top of a sheet that ends below them,
    and both used to light Data. Each tab, clicked in turn, lights itself and is aria-current."""
    ctx, page, errors = _open(browser)
    try:
        _sheet(page, "APA")
        tabs = page.evaluate("[...document.querySelectorAll('#stock-modal .modal-nav a')].filter(a => !a.hidden).map(a => a.getAttribute('href').slice(1))")
        assert "section-snapshot" in tabs
        for t in tabs:
            page.click(f"#stock-modal .modal-nav a[href='#{t}']")
            page.wait_for_function(f"""() => {{ const a = document.querySelector("#stock-modal .modal-nav a[href='#{t}']");
                return a.classList.contains('active') && a.getAttribute('aria-current') === 'location' &&
                       document.querySelectorAll('#stock-modal .modal-nav a.active').length === 1; }}""", timeout=WAIT)
        # still lit once the smooth scroll has settled at the end of the sheet
        page.wait_for_function("""() => { const b = document.querySelector('#stock-modal .modal-body');
            return b.scrollTop + b.clientHeight >= b.scrollHeight - 2; }""", timeout=WAIT)
        assert page.evaluate(LIT) == "section-provenance"
        page.click("#stock-modal .modal-nav a[href='#section-peers']")
        page.wait_for_function("""() => { const b = document.querySelector('#stock-modal .modal-body');
            return b.scrollTop + b.clientHeight >= b.scrollHeight - 2; }""", timeout=WAIT)
        assert page.evaluate(LIT) == "section-peers"
        # the reader scrolling by hand releases it: back at the top, Scores is lit
        page.mouse.move(900, 500)
        page.mouse.wheel(0, -400)
        page.wait_for_function("SPY_PIN === null", timeout=WAIT)
        page.evaluate("document.querySelector('#stock-modal .modal-body').scrollTop = 0")
        page.wait_for_function(f"{LIT} === 'modal-score-row'", timeout=WAIT)
        assert errors == []
    finally:
        ctx.close()


def test_scrolling_freely_lights_every_tab_in_order(browser):
    """With every section open, stepping down the sheet lights the shown tabs in reading order -
    none skipped, none out of order - and the last one at the bottom."""
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        _sheet(page, "JPM")
        page.evaluate("document.querySelectorAll('#stock-modal .collapsible.collapsed').forEach(s => s.classList.remove('collapsed'))")
        tabs = page.evaluate("[...document.querySelectorAll('#stock-modal .modal-nav a')].filter(a => !a.hidden).map(a => a.getAttribute('href').slice(1))")
        seen = page.evaluate("""async () => {
            const b = document.querySelector('#stock-modal .modal-body');
            const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
            const out = [];
            for (let y = 0; ; y += 60) {
                b.scrollTop = y;
                await frame();
                const a = document.querySelector('#stock-modal .modal-nav a.active');
                const id = a && a.getAttribute('href').slice(1);
                if (id && out[out.length - 1] !== id) out.push(id);
                if (b.scrollTop + b.clientHeight >= b.scrollHeight - 1) break;
            }
            return out; }""")
        assert seen == tabs, (seen, tabs)
        assert errors == []
    finally:
        ctx.close()


WARM_DONE = "typeof SHEET_WARM !== 'undefined' && SHEET_WARM.state === 'done'"


def test_the_first_sheet_is_laid_out_unseen_and_leaves_nothing_behind(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        before = page.evaluate("({ title: document.title, url: location.href, hist: history.length, focus: document.activeElement === document.body })")
        page.wait_for_function(WARM_DONE, timeout=WAIT)
        st = page.evaluate("""() => ({ ticker: SHEET_WARM.ticker, open: sheetOpen(), display: document.getElementById('stock-modal').style.display,
            warm: document.querySelectorAll('#stock-modal.sheet-warm, #stock-modal .warm-skip').length,
            filled: document.getElementById('modal-ticker').textContent,
            title: document.title, url: location.href, hist: history.length, focus: document.activeElement === document.body,
            overflow: document.body.style.overflow })""")
        top = page.evaluate("D.table_data.slice().sort((a, b) => a.Rank - b.Rank)[0].Ticker")
        assert st["ticker"] == top and st["filled"] == top          # it did lay out the top stock
        assert st["open"] is False and st["display"] == "none" and st["warm"] == 0
        assert (st["title"], st["url"], st["hist"], st["focus"]) == (before["title"], before["url"], before["hist"], before["focus"])
        assert st["overflow"] == ""
        # a real open after it is an ordinary open, of the stock asked for
        _sheet(page, "JPM")
        assert page.evaluate("document.getElementById('modal-ticker').textContent") == "JPM"
        assert page.evaluate("getComputedStyle(document.getElementById('stock-modal')).visibility") == "visible"
        assert errors == []
    finally:
        ctx.close()


def test_opening_a_stock_first_ends_the_warm_up(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True, hash_="#stock=JPM")
    try:
        page.wait_for_function("document.getElementById('stock-modal').style.display === 'flex'", timeout=WAIT)
        page.wait_for_function(WARM_DONE, timeout=WAIT)
        st = page.evaluate("""() => ({ t: document.getElementById('modal-ticker').textContent,
            warm: document.querySelectorAll('#stock-modal.sheet-warm, #stock-modal .warm-skip').length,
            vis: getComputedStyle(document.getElementById('stock-modal')).visibility })""")
        assert st == {"t": "JPM", "warm": 0, "vis": "visible"}
        assert errors == []
    finally:
        ctx.close()


def test_an_unscored_tile_names_its_metrics_and_a_rejected_price_history(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        t, cats = page.evaluate("""() => { const C = ['valuation','quality','growth','momentum','risk','revisions','size','investment'];
            for (const t of Object.keys(D.stock_detail)) { const n = C.filter(c => D.stock_detail[t].cat_scores[c] == null);
                if (n.length) return [t, n]; } return [null, []]; }""")
        assert t, "no unscored category in this run"
        _sheet(page, t)
        subs = page.evaluate("""() => [...document.querySelectorAll('#stock-modal .modal-score-card.is-na .modal-score-sub')].map(e => e.textContent)""")
        assert len(subs) == len(cats)
        assert all(re.fullmatch(r"Not scored: (none of its \d+ metrics has data|its one metric has no data)", x) for x in subs), subs
        # the workings name the metrics, by the published labels
        c = cats[0]
        note = page.evaluate(f"document.querySelector('#cat-detail-{c} .wk-unscored').textContent")
        labels = page.evaluate(f"unscoredWhy('{c}', D.stock_detail['{t}']).metrics")
        assert labels and all(lbl in note for lbl in labels), (note, labels)
        assert "its weight is shared across the others" in note
        # a rejected price history (no stock has one today): Momentum and Risk give the engine's reason
        page.evaluate("""() => { const s = D.stock_detail.JPM; s.__keep = [s.cat_scores.momentum, s.cat_scores.risk, s.flags.px_rejected];
            s.cat_scores.momentum = null; s.cat_scores.risk = null; s.flags.px_rejected = 'series mixes pre- and post-split prices across a 2:1 split'; }""")
        _sheet(page, "JPM")
        subs = page.evaluate("""() => [...document.querySelectorAll('#stock-modal .modal-score-card.is-na .modal-score-sub')].map(e => e.textContent)""")
        assert subs == ["Not scored: price history rejected"] * 2, subs
        note = page.evaluate("document.querySelector('#cat-detail-risk .wk-unscored').textContent")
        assert "rejected this stock’s daily price history (series mixes pre- and post-split prices across a 2:1 split)" in note
        # Quality is not built from the price history and never claims it
        assert "price history" not in page.evaluate("unscoredWhy('quality', D.stock_detail.JPM).long")
        assert errors == []
    finally:
        ctx.close()


def test_empty_sections_say_so(browser):
    ctx, page, errors = _open(browser)
    try:
        page.evaluate("""() => { const s = D.stock_detail.JPM; s.pt_mean = s.pt_high = s.pt_low = null; s.financials = null; s.self_metrics = null; }""")
        _sheet(page, "JPM")
        got = page.evaluate("['modal-price-targets', 'modal-snapshot', 'modal-peers'].map(id => document.getElementById(id).textContent.trim())")
        assert got == ["No analyst price target was published for this stock in this run.",
                       "No financial figures were published for this stock in this run.",
                       "No peer figures were published for this stock in this run."]
        assert errors == []
    finally:
        ctx.close()


def test_a_peer_opens_its_own_sheet_and_shows_its_rank(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        _sheet(page, "JPM")
        page.click("#stock-modal .modal-nav a[href='#section-peers']")
        page.wait_for_selector("#modal-peers .peer-open", state="visible", timeout=WAIT)
        rows = page.evaluate("""() => [...document.querySelectorAll('#modal-peers tbody tr:not(.peer-row-median)')].map(r => [
            r.querySelector('.peer-ticker').textContent.trim(), r.querySelector('.peer-td-rank').textContent.trim(),
            (r.querySelector('.peer-remove-btn') || {}).ariaLabel || null])""")
        D = _payload()
        for tick, rank, rm in rows:
            assert rank == "#%d" % D["stock_detail"][tick]["rank"], (tick, rank)
            assert (rm is None) == (tick == "JPM")
            assert rm is None or rm == f"Remove {tick} from the comparison"
        assert page.evaluate("document.querySelector('#modal-peers th.peer-th-action').textContent.trim()") == "Remove"
        # the peer's ticker is a real button, 44 px on a touch screen, and opens that stock
        peer = rows[1][0]
        assert page.evaluate("document.querySelector('#modal-peers .peer-open').getBoundingClientRect().height") >= 44
        page.click(f"#modal-peers .peer-open >> text='{peer}'")
        page.wait_for_function(f"document.getElementById('modal-ticker').textContent === {json.dumps(peer)}", timeout=WAIT)
        assert page.evaluate("sheetOpen()")
        assert errors == []
    finally:
        ctx.close()
