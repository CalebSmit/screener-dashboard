"""Investment-club readiness: can a first-year student open this on a phone and understand it?

WHY THIS EXISTS - 2026-10-19 (CLAUDE.md priority 7, ``plan/club-readiness.md``).

A walkthrough at 375px as a first-year finance student found the page explained itself to a
mouse and not to a thumb. The table's header tooltips said which way each score runs ("a high
Risk score means LOW risk") and what the trap-flag column was; a phone hides that header and a
touch screen never shows a tooltip. So a phone reader met "Risk 21", "Size 2" and a bare "Value"
chip with nothing on the page to say what they meant, and searching for a stock that has left the
index answered "Try a wider sector".

The fixes are display-only: a "What the columns mean" key above the ranking (also reached from
the Trap flags tile and the search palette), one line under the stock sheet's scores, the guide's
first statement, "trap" on the phone's flag chips, a named answer for a ticker this run did not
score, a gloss for "pp" in the Market backdrop, and a two-stock comparison that fits a phone.

The key says how flags are computed and which way each score runs, so it is held to the rule for
every such sentence (``claims.py``, ``dashboard.score_key`` and ``dashboard.trap_flag_key``): the
thresholds come from ``config.yaml``, the engine must flag what the key describes, and each
category's score must move with what its line says on the published payload.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import claims  # noqa: E402
import factor_engine  # noqa: E402
import generate_dashboard as gd  # noqa: E402
import stock_summary  # noqa: E402

PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


@pytest.fixture(scope="module")
def html() -> str:
    return gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip("no published payload")
    s = PAYLOAD.read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))


def _key(html: str) -> str:
    return html[html.index('id="score-key"'):html.index("</details>", html.index('id="score-key"'))]


# ---------------------------------------------------------------------------
# the key's markup
# ---------------------------------------------------------------------------

def test_the_key_sits_above_the_ranking_and_names_every_column(html):
    assert html.index('id="score-key"') < html.index('id="universe-table"')
    key = _text(_key(html))
    assert "Composite" in key and "Value trap" in key and "Growth trap" in key
    for cat, short, name, _ in gd.SCORE_KEY:
        assert name in key and short in key, cat
    assert [c for c, *_ in gd.SCORE_KEY] == CATS
    # the run-specific lines are filled from the payload, never typed in
    assert 'id="sk-median"' in html and 'id="sk-delta"' in html
    assert "k.median_composite" in html and "H.noise.material_threshold" in html


def test_three_ways_in_to_the_key(html):
    """The Trap flags tile, the guide and the search palette all open it."""
    assert "openScoreKey(\\'key-traps\\')" in html          # the tile's link, inside a JS string
    guide = html[html.index('<section class="guide"'):html.index("<!-- Top 5 Stocks -->")]
    assert "openScoreKey()" in guide
    assert "label: 'What the columns mean'" in gd._js_ux()
    assert "a.keys" in gd._js_ux()                          # "trap" finds it in the palette


def test_the_key_reads_the_trap_thresholds_from_config(cfg):
    key = _text(gd.score_key_html(cfg))
    v, g = cfg["value_trap_filters"], cfg["growth_trap_filters"]
    assert f"top {100 - v['valuation_percentile']}% of Valuation scores" in key
    assert f"bottom {v['quality_floor_percentile']}% on at least two of Quality, Momentum and Revisions" in key
    assert f"top {100 - g['growth_ceiling_percentile']}% of Growth scores" in key
    assert f"Quality or Revisions in the bottom {g['quality_floor_percentile']}%" in key
    # and it follows the config when the config moves
    moved = {"value_trap_filters": {**v, "valuation_percentile": 80},
             "growth_trap_filters": {**g, "growth_ceiling_percentile": 75}}
    key2 = _text(gd.score_key_html(moved))
    assert "top 20% of Valuation scores" in key2 and "top 25% of Growth scores" in key2


def test_the_engine_flags_what_the_key_describes(cfg):
    """Value trap: high Valuation AND weak on at least two of three. Growth trap: high Growth AND
    weak Quality or Revisions. Built so each stock tests one clause of the key's sentence."""
    n = 100
    base = {c + "_score": [50.0] * n for c in CATS}
    df = pd.DataFrame(base)
    # stock 0: top Valuation, bottom on Quality and Momentum -> value trap
    # stock 1: top Valuation, bottom on Quality only -> not a value trap
    # stock 2: bottom Valuation, bottom on all three -> not a value trap (it must be priced low)
    df.loc[0, ["valuation_score", "quality_score", "momentum_score"]] = [99, 1, 1]
    df.loc[1, ["valuation_score", "quality_score"]] = [98, 2]
    df.loc[2, ["valuation_score", "quality_score", "momentum_score", "revisions_score"]] = [1, 3, 3, 3]
    # stock 3: top Growth, bottom Revisions -> growth trap; stock 4: low Growth, weak both -> not
    df.loc[3, ["growth_score", "revisions_score"]] = [99, 1]
    df.loc[4, ["growth_score", "quality_score", "revisions_score"]] = [1, 2, 2]
    # spread the rest so the percentile cut-offs are well defined
    for c in ("valuation_score", "growth_score", "quality_score", "momentum_score", "revisions_score"):
        df.loc[5:, c] = [10 + (i * 7) % 80 for i in range(n - 5)]
    df = factor_engine.apply_value_trap_flags(df, cfg)
    df = factor_engine.apply_growth_trap_flags(df, cfg)
    assert df.loc[0, "Value_Trap_Flag"] and not df.loc[1, "Value_Trap_Flag"] and not df.loc[2, "Value_Trap_Flag"]
    assert df.loc[3, "Growth_Trap_Flag"] and not df.loc[4, "Growth_Trap_Flag"]


# ---------------------------------------------------------------------------
# the key's sentences against the published payload
# ---------------------------------------------------------------------------

# One weighted metric per category whose direction the key's line names, and the sign a higher
# category score must carry against it.
DIRECTION = {
    "valuation": ("fcf_yield", +1, "lower price"),
    "quality": ("roic", +1, "more profitable"),
    "growth": ("forward_eps_growth", +1, "faster growth"),
    "momentum": ("return_12_1", +1, "stronger price trend"),
    "risk": ("volatility", -1, "lower volatility"),
    "revisions": ("fy1_revision_3m", +1, "expectations are improving"),
    "size": ("market_cap", -1, "smaller company"),
    "investment": ("asset_growth", -1, "slower growth in total assets"),
}


def test_each_line_of_the_key_says_which_way_its_score_runs(payload):
    lines = {cat: text for cat, _, _, text in gd.SCORE_KEY}
    for cat, (metric, sign, words) in DIRECTION.items():
        assert words in lines[cat], (cat, lines[cat])
        rows = []
        for v in payload["stock_detail"].values():
            x = v["financials"].get("market_cap") if metric == "market_cap" else v["raw"].get(metric)
            rows.append((v["cat_scores"].get(cat), x))
        d = pd.DataFrame(rows, columns=["s", "x"]).dropna()
        rho = d["s"].corr(d["x"], method="spearman")
        assert len(d) > 300 and sign * rho > 0.5, (cat, metric, round(rho, 3), len(d))


def test_the_largest_company_in_each_sector_scores_near_zero_on_size(payload):
    by_sector: dict[str, list] = {}
    for t, v in payload["stock_detail"].items():
        mc, sz = v["financials"].get("market_cap"), v["cat_scores"].get("size")
        if mc and sz is not None:
            by_sector.setdefault(v["sector"], []).append((mc, t, sz))
    assert len(by_sector) >= 10
    for sector, rows in by_sector.items():
        mc, t, sz = max(rows)
        assert sz < 5, (sector, t, sz)


def test_a_flag_changes_neither_the_score_nor_the_rank(payload):
    rows = sorted(payload["table_data"], key=lambda r: r["Rank"])
    comps = [r["Composite"] for r in rows]
    assert all(a >= b for a, b in zip(comps, comps[1:])), "rank is not the composite order"
    flagged = [r for r in rows if r["Value_Trap_Flag"] or r["Growth_Trap_Flag"]]
    assert flagged and min(r["Rank"] for r in flagged) < 50     # flagged names keep their places
    js = gd.generate_html()
    top5 = js[js.index("function renderTop5()"):js.index("function renderTop5()") + 1500]
    assert "Value_Trap_Flag" in top5 and "Growth_Trap_Flag" in top5


def test_the_median_the_key_quotes_is_the_engines(payload):
    k = payload["kpis"]
    comps = sorted(r["Composite"] for r in payload["table_data"])
    assert isinstance(k["median_composite"], (int, float))
    assert abs(k["median_composite"] - pd.Series(comps).median()) < 0.06


def test_the_claims_are_registered():
    ids = {c.id for c in claims.CLAIMS}
    assert {"dashboard.score_key", "dashboard.trap_flag_key"} <= ids


# ---------------------------------------------------------------------------
# the copy
# ---------------------------------------------------------------------------

def _new_copy(html: str) -> dict[str, str]:
    js_table, js_ux, js_ctx = gd._js_table(), gd._js_ux(), gd._js_context()
    note = re.search(r"msr-note\">([^']+)</p>", html).group(1)
    gone = re.search(r"return 'The ranking covers[^\n]+", js_table).group(0)
    market = re.search(r"The macro backdrop for this run[^<]+", js_ctx).group(0)
    guide = html[html.index('<section class="guide"'):html.index("<!-- Top 5 Stocks -->")]
    return {"key": _text(_key(html)), "sheet note": _text(note), "not in run": gone,
            "market": market, "guide": _text(guide)}


def test_the_new_copy_carries_no_advice_language(html):
    for name, text in _new_copy(html).items():
        assert stock_summary.advice_terms_in(text) == [], (name, stock_summary.advice_terms_in(text))


def test_the_new_copy_is_plain_declarative(html):
    for name, text in _new_copy(html).items():
        assert "—" not in text and "&mdash;" not in text, name
        assert not re.search(r"\w, not (a|an|the)?\s?\w+", text), (name, text)


def test_the_tested_page_says_how_to_read_its_numbers():
    """research.html opens on a rank correlation of -0.003 with t-statistics and basis points
    below it; a first-year reader needs the scale. The page is generated, and
    tests/test_research_page.py holds the committed file to the generator."""
    page = (ROOT / "research.html").read_text(encoding="utf-8")
    terms = page[page.index('class="fine terms"'):]
    terms = _text(terms[:terms.index("</p>")])
    for phrase in ("A rank correlation runs from", "A t-statistic is", "A basis point is"):
        assert phrase in terms, phrase
    assert page.index('class="fine terms"') < page.index("How it was tested")
    assert stock_summary.advice_terms_in(terms) == []


def test_the_market_backdrop_glosses_pp_and_fred():
    js = gd._js_context()
    assert "A pp is a percentage point" in js
    assert "St. Louis Fed" in js


# ---------------------------------------------------------------------------
# in a browser, on a phone
# ---------------------------------------------------------------------------

sync_api = None
try:  # pragma: no cover - environment dependent
    from playwright import sync_api  # type: ignore
except Exception:  # noqa: BLE001
    pass

needs_browser = pytest.mark.skipif(sync_api is None or not PAGE.exists() or not PAYLOAD.exists(),
                                   reason="playwright or the generated dashboard is not present")


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _open(browser, phone=True, hash_=""):
    if phone:
        ctx = browser.new_context(viewport={"width": 375, "height": 812}, is_mobile=True, has_touch=True)
    else:
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri() + hash_, wait_until="load", timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
    return ctx, page, errors


def _no_sideways_scroll(page):
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@needs_browser
def test_a_phone_reader_reaches_the_flag_definitions_from_the_tile(browser):
    ctx, page, errors = _open(browser)
    try:
        page.tap("#kpi-row a.kpi-more")
        page.wait_for_function("document.getElementById('score-key').open")
        page.wait_for_timeout(700)
        box = page.locator("#key-traps").bounding_box()
        assert box and 0 <= box["y"] < 812, box
        delta = page.inner_text("#sk-delta")
        h = page.evaluate("({n: H.noise.material_threshold, d: H.compare[rankDeltaKey()].date})")
        assert f"Moves under {h['n']} places" in delta
        assert "50." in page.inner_text("#sk-median") or "median this run" in page.inner_text("#sk-median")
        assert _no_sideways_scroll(page)
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_a_ticker_this_run_did_not_score_is_named(browser):
    ctx, page, errors = _open(browser)
    try:
        page.evaluate("openPalette()")
        page.keyboard.type("WBA")
        assert "WBA is not in this ranking." in page.inner_text("#pal-list")
        page.keyboard.press("Escape")
        page.fill("#filter-search", "WBA")
        page.wait_for_selector(".vt-empty")
        assert "WBA is not in this ranking." in page.inner_text(".vt-empty")
        # a name fragment that matches nothing keeps the ordinary message
        page.fill("#filter-search", "zzzz corp")
        page.wait_for_timeout(200)
        assert "No stocks match these filters." in page.inner_text(".vt-empty")
        # a scored ticker hidden by another filter is not called missing
        page.fill("#filter-search", "AAPL")
        page.select_option("#filter-sector", "Energy")
        page.wait_for_timeout(200)
        assert "No stocks match these filters." in page.inner_text(".vt-empty")
        # and the My holdings add box, where a student first types what they own
        page.evaluate("goToSection('sec-holdings')")
        page.wait_for_timeout(400)
        page.fill("#holdings-search-input", "WBA")
        page.wait_for_selector("#holdings-search-results .peer-search-empty", state="visible")
        assert "WBA is not in this ranking." in page.inner_text("#holdings-search-results")
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_flag_chips_say_trap_where_the_header_is_hidden(browser):
    """Drawn with CSS on a phone only, so the row's text (and test_dashboard_browser's check that
    flags are the words Value / Growth) is unchanged."""
    shown = ("(() => { const f = document.querySelector('#universe-tbody .flag');"
             " const a = getComputedStyle(f, '::after').content;"
             " return f.textContent + (a === 'none' ? '' : a.replace(/\"/g, '')); })()")
    for phone in (True, False):
        ctx, page, _ = _open(browser, phone=phone)
        try:
            page.select_option("#filter-vt", "vt")
            page.wait_for_timeout(200)
            chip = page.evaluate(shown)
            assert chip == ("Value trap" if phone else "Value"), chip
        finally:
            ctx.close()


@needs_browser
def test_the_sheet_says_which_way_the_scores_run(browser):
    ctx, page, errors = _open(browser, hash_="#stock=JPM")
    try:
        page.wait_for_selector("#stock-modal .msr-note", state="visible")
        note = page.inner_text("#stock-modal .msr-note")
        assert "a high Size score a smaller company" in note and "a high Risk score means a calmer price" in note
        assert _no_sideways_scroll(page)
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_two_stocks_side_by_side_fit_a_phone(browser):
    ctx, page, errors = _open(browser)
    try:
        page.evaluate("try { localStorage.setItem('screener_ux_compare', JSON.stringify(['EXPE', 'JPM'])) } catch (e) {}")
        page.reload(wait_until="load")
        page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
        page.evaluate("openCompare()")
        page.wait_for_selector(".cmp-scroll", state="visible")
        fits = page.evaluate("(() => { const s = document.querySelector('.cmp-scroll'); return s.scrollWidth <= s.clientWidth + 1; })()")
        assert fits
        assert errors == []
    finally:
        ctx.close()
