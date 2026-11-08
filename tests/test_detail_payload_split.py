"""The third payload file: the sheet-only half of ``stock_detail``, loaded on demand.

2026-11-08, the first slice of `plan/universe-expansion.md` phase 1's last bullet - shard
`stock_detail` so that first paint does not grow with the universe.

What was measured before the split
(`research/measurements/2026-11-08-stock-detail-field-weights.py`): `stock_detail` is **84.3%**
of the 1.547 MB gzipped first-paint payload, and the single heaviest field in the whole payload
is **`about`, 19.3%** - the provider's business description, ahead of `inp` (13.8%) and `raw`
(12.5%). It is display-only prose by the 2026-08-26 decision, nothing scores or ranks it, and
**one** place in the page reads it (`renderAbout`, inside the open sheet). So it ships in
``dashboard_detail.js`` with `industry` beside it, fetched the first time a sheet is laid out.

**Measured, because the obvious claim is wrong:** that is not "only if a reader opens a stock".
The idle-time pre-warm lays the top stock's sheet out on a page nobody touches, so it pulls the
file too - after the payload and after the context layer, which is the order the requests
actually arrive in (`test_it_is_the_last_of_the_three_payloads_to_be_requested`). First paint is
what the split protects, and the pre-warm is there so that the first open is instant.

**`financials` joined the file the same day (13:00 session), and with it the rule was
restated.** It was written that morning as "one read site, inside the sheet, and no number",
which described what made `about` safe rather than what the shard can carry. It is now:

    a field may move when every surface that reads it is the open sheet, and every panel that
    prints it can say honestly that it is waiting.

`financials` passes both, and the second half is why the first *numbers* in the file needed a
mechanism rather than a tuple entry: the loader now redraws the Company snapshot and the peer
table when the file lands, and both say they are waiting until it does - the snapshot in words,
the peer table's three `financials`-fed cells with an ellipsis rather than the em-dash that in a
column of figures means "this run published none". It also only became true today that the sheet
is the *only* reader: until the peer table was unified onto `buildPeerRow`, the subject's row
came from a precomputed `self_metrics` block (`tests/test_peer_row_one_path.py`).

Still outside, and why: `inp`, `raw`, `pct` and `summary` are 42.6% between them and are read by
the movers strip, the compare drawer and the calculation traces - surfaces that are not the sheet
and are live at first paint. Each needs its own read-site audit before it moves.

The tests are in three groups:

1. **the split itself**, against the generator rather than the committed file (CLAUDE.md rule
   10: a test that reads a generated artifact passes on a hand-edit and says nothing about what
   the next run will publish);
2. **the page**, which must not fetch the file before first paint and must say which state the
   block is in while it waits;
3. **the published artifacts**, as relationships only - never which stock today's fetch
   happened to describe (settled row "data->red").
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

PAYLOAD = ROOT / "dashboard_data.js"
DETAIL = ROOT / "dashboard_detail.js"
CATEGORIES = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


# --------------------------------------------------------------------------- 1. the split itself
def _frame() -> pd.DataFrame:
    """Four stocks, and the three ways a description arrives: text, NaN, empty string."""
    df = pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC", "DDD"],
        "Company": ["Alpha Inc", "Beta Corp", "Gamma Ltd", "Delta SA"],
        "Sector": ["Information Technology", "Financials", "Energy", "Utilities"],
        "Composite": [80.0, 60.0, 55.0, 40.0],
        "Rank": [1, 2, 3, 4],
        "Value_Trap_Flag": [False, False, False, False],
        "Growth_Trap_Flag": [False, False, False, False],
        "_about": ["Alpha Inc designs and sells things worldwide.", np.nan,
                   "Gamma Ltd explores for and produces crude oil.", ""],
        "_industry": ["Software - Infrastructure", "Banks - Diversified", "Oil & Gas E&P", ""],
    })
    for cat in CATEGORIES:
        df[cat + "_score"] = 50.0
    return df


@pytest.fixture(scope="module")
def built() -> tuple[dict, dict]:
    """The payload as the generator builds it, then split: (first paint, on demand)."""
    data = json.loads(gd.prepare_dashboard_data({
        "df": _frame(), "meta": {"run_date": "2026-11-08"},
        "weights": {}, "sens_df": None, "corr_df": None, "cfg": {},
    }))
    detail = gd.split_detail(data)
    return data, detail


def test_the_fields_leave_the_first_paint_payload(built):
    data, _ = built
    for ticker, s in data["stock_detail"].items():
        for field in gd.DETAIL_KEYS:
            assert field not in s, f"{ticker} still carries {field} in the first-paint payload"


def test_the_fields_arrive_in_the_detail_payload(built):
    _, detail = built
    assert detail["detail"]["AAA"] == {"about": "Alpha Inc designs and sells things worldwide.",
                                       "industry": "Software - Infrastructure"}
    assert detail["detail"]["CCC"]["about"].startswith("Gamma Ltd explores")


def test_a_stock_with_nothing_to_say_is_omitted_rather_than_carried_empty(built):
    """``DDD`` has an empty description and an empty industry; ``BBB`` has a NaN description and
    an industry, so it is carried - by its industry alone. Empty strings for 500 stocks would be
    ~10 KB of nothing, and since 2026-11-08 (13:00) the emptiness is judged **per key**: a stock
    with a description and no figures carries the description and not an empty block beside it."""
    _, detail = built
    assert "DDD" not in detail["detail"]
    assert detail["detail"]["BBB"] == {"industry": "Banks - Diversified"}


def test_the_split_is_in_place_so_the_two_halves_cannot_both_be_published(built):
    """``split_detail`` pops. If it ever copied instead, first paint would carry the weight the
    split exists to remove while the page fetched a second copy of it."""
    data, detail = built
    carried = {t: [f for f in gd.DETAIL_KEYS if f in s] for t, s in data["stock_detail"].items()}
    assert not any(carried.values()), carried
    assert detail["detail"], "the split moved nothing at all"


def test_the_detail_fields_are_never_scored(built):
    """The invariant that matters, and it covers every key: a field in this file is display-only.

    It is **not** "the file holds no numbers" - `financials` joined ``DETAIL_KEYS`` on 2026-11-08
    (13:00) and is a block of figures. What makes a field safe to move is that no scoring path
    reads it and no surface outside the open sheet prints it; `financials` is the Company
    snapshot's and the peer table's, both inside the sheet. Being a *string* is a property of the
    provider prose only, which the next test keeps."""
    import factor_engine as fe
    for field in gd.DETAIL_KEYS:
        assert field not in fe.METRIC_COLS
    data, detail = built
    for s in data["stock_detail"].values():
        for field in gd.DETAIL_KEYS:
            assert field not in (s.get("raw") or {}) and field not in (s.get("pct") or {})


def test_the_prose_fields_are_prose(built):
    """The 2026-08-26 decision, kept on the subset it is about. ``PROSE_KEYS`` is the provider's
    own text; a number appearing there would mean a metric had been routed into prose."""
    assert set(gd.PROSE_KEYS) <= set(gd.DETAIL_KEYS)
    _, detail = built
    for part in detail["detail"].values():
        for k in gd.PROSE_KEYS:
            if k in part:
                assert isinstance(part[k], str), f"{k} is not prose"


def test_a_block_of_figures_with_no_figure_in_it_is_not_carried(built):
    """``_has_content`` rather than ``str(v or "").strip()``, which is truthy for *every* dict -
    including one whose every value is None. Without it the day `financials` joined
    ``DETAIL_KEYS`` the file would have gained ~500 entries of nothing."""
    assert gd._has_content({"a": 1}) and gd._has_content("x") and gd._has_content(0.0)
    assert not gd._has_content({"a": None, "b": None})
    assert not gd._has_content({}) and not gd._has_content("  ") and not gd._has_content(None)
    _, detail = built
    for t, part in detail["detail"].items():
        assert any(gd._has_content(v) for v in part.values()), t


def test_the_context_split_is_untouched(built):
    """Two splits, two files. `split_detail` must not take the context layer with it, and the
    `ctx` key stays where `plan/context-layer.md` says it is."""
    _, detail = built
    assert set(detail) == {"detail"}
    for k in gd.CONTEXT_KEYS:
        assert k not in detail


# --------------------------------------------------------------------------- 2. the page
@pytest.fixture(scope="module")
def html() -> str:
    return gd.generate_html(data_version="dv", context_version="cv", detail_version="xv")


def test_the_page_names_the_file_with_its_own_version(html):
    assert "dashboard_detail.js?v=xv" in html
    assert "window.SCREENER_DETAIL" in html


def test_the_version_falls_back_to_the_payload_version_rather_than_to_nothing():
    """A cache-busting token of the empty string would serve a stale description forever."""
    h = gd.generate_html(data_version="dv")
    assert "dashboard_detail.js?v=dv" in h
    assert "dashboard_detail.js?v=" not in h.replace("dashboard_detail.js?v=dv", "")


def test_the_file_is_not_fetched_before_first_paint(html):
    """The whole point. The context layer is loaded from the page's start-up block; this one is
    loaded from ``renderAbout``, which runs only when a sheet is laid out - so a reader who never
    opens a stock never pays for it, and first paint does not carry it either way.
    """
    start = html.index("renderKPIs();")
    block = html[start:html.index("</script>", start)]
    assert "loadContext(" in block, "the start-up block is no longer where the context is loaded"
    assert "loadDetail(" not in block, (
        "dashboard_detail.js is fetched from the page's start-up block - that puts the weight "
        "back on the critical path, which is the thing the split removed")
    about = html[html.index("function renderAbout"):]
    assert "loadDetail(" in about[:about.index("function toggleAbout")]


def test_the_loader_merges_into_stock_detail_so_existing_readers_need_no_change(html):
    loader = html[html.index("function loadDetail"):]
    loader = loader[:loader.index("function loadContext")]
    assert "Object.assign(D.stock_detail[t], P[t])" in loader
    assert "DETAIL_LOADED = true" in loader


def test_the_loader_cannot_request_the_file_twice(html):
    """``renderAbout`` runs on every sheet, including the idle-time pre-warm."""
    loader = html[html.index("function loadDetail"):]
    loader = loader[:loader.index("function loadContext")]
    assert "if (DETAIL_LOADED || DETAIL_PENDING) return;" in loader


def test_the_block_says_which_state_it_is_in_while_it_waits(html):
    """An empty gap where the description goes reads as "this company has no description", which
    is true of approximately none of them. Both waiting states are named."""
    about = html[html.index("function renderAbout"):]
    about = about[:about.index("function toggleAbout")]
    assert "Loading the company description" in about
    assert "could not be loaded" in about
    assert "DETAIL_FAILED" in about


# --------------------------------------------------------------------------- 3. what is published
def _published(path: Path, prefix: str) -> dict:
    txt = path.read_text(encoding="utf-8")
    assert txt.startswith(prefix), f"{path.name}: unexpected framing"
    return json.loads(txt[len(prefix):].rstrip().rstrip(";"))


@pytest.fixture(scope="module")
def live() -> tuple[dict, dict]:
    for p in (PAYLOAD, DETAIL):
        if not p.exists():
            pytest.skip(f"MISSING DATA: no published {p.name}")
    return (_published(PAYLOAD, "window.SCREENER_DATA = "),
            _published(DETAIL, "window.SCREENER_DETAIL = "))


def test_the_published_payload_carries_none_of_the_detail_fields(live):
    data, _ = live
    offenders = [t for t, s in data["stock_detail"].items()
                 if any(f in s for f in gd.DETAIL_KEYS)]
    assert not offenders, f"{len(offenders)} published stocks still carry the sheet-only fields"


def test_every_described_stock_is_one_the_page_can_open(live):
    """No orphans: a description for a ticker the ranking does not carry is weight the page can
    never show. A relationship, not a count - the universe changes every quarter."""
    data, detail = live
    missing = sorted(set(detail["detail"]) - set(data["stock_detail"]))
    assert not missing, f"described but not published: {missing[:10]}"


def test_the_two_files_do_not_describe_the_same_field_twice(live):
    data, detail = live
    for t, part in detail["detail"].items():
        overlap = set(part) & set(data["stock_detail"][t])
        assert not overlap, f"{t}: {sorted(overlap)} published in both files"


def test_the_published_detail_file_is_framed_the_way_both_runners_check_for(live):
    """Both publish paths check the opening assignment and run ``node --check``; the regex they
    use is anchored, so a leading space or a renamed global would be refused at 02:00."""
    first = DETAIL.read_text(encoding="utf-8")[:40].replace(" ", "")
    assert re.match(r"^window\.SCREENER_DETAIL=", first)


def test_the_published_page_asks_for_the_version_it_published(live):
    """The cache-busting token on the live page must be the hash of the live file.

    Without this, a build that forgot to thread `detail_version` would publish a page asking for
    ``?v=<the payload's hash>`` - which is a URL a returning reader already has cached, so they
    would be served yesterday's descriptions indefinitely. The same check `scripts/rebuild_page.py`
    makes before it rewrites the page; here it is a test, because a rebuild is not run every day.
    """
    body = DETAIL.read_text(encoding="utf-8")[len("window.SCREENER_DETAIL = "):-1]
    want = gd.hashlib.md5(body.encode("utf-8")).hexdigest()[:12]
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    assert f"dashboard_detail.js?v={want}" in page, (
        "index.html does not ask for the committed detail file's version")


def test_the_file_is_one_the_data_loop_publishes_and_both_runners_guard():
    """The 2026-10-16 lesson, applied in the commit that creates the file rather than eight days
    later. `tests/test_payload_parse_gate.py` generalises it; this is the direct statement."""
    for name in ("scripts/data-run.ps1", "scripts/nightly-screener.ps1"):
        assert "dashboard_detail.js" in (ROOT / name).read_text(encoding="utf-8"), name
    claims_gate = (ROOT / "scripts" / "check_published_claims.py").read_text(encoding="utf-8")
    assert "dashboard_detail.js" in claims_gate


# --------------------------------------------------------------------------- 4. end to end
@pytest.fixture(scope="module")
def page_and_ticker(tmp_path_factory):
    """The page as the site serves it, from a copy, so a held route is possible."""
    if sync_api is None:
        pytest.skip("playwright not installed")
    for p in (PAYLOAD, DETAIL):
        if not p.exists():
            pytest.skip(f"MISSING DATA: no published {p.name}")
    data = _published(PAYLOAD, "window.SCREENER_DATA = ")
    detail = _published(DETAIL, "window.SCREENER_DETAIL = ")
    # Derived from the payload, never a named stock (settled row "data->red"): the best-ranked
    # stock that has a description to show.
    ranked = sorted((r for r in data["table_data"] if r.get("Rank")), key=lambda r: (r["Rank"], r["Ticker"]))
    ticker = next((r["Ticker"] for r in ranked if (detail["detail"].get(r["Ticker"]) or {}).get("about")), None)
    if ticker is None:
        pytest.skip("MISSING DATA: no ranked stock carries a description")
    d = tmp_path_factory.mktemp("detail")
    version = gd.hashlib.md5(DETAIL.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:12]
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v", detail_version=version),
                                  encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as pw:
        try:
            b = pw.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")

        def make(hold: bool = False, watch: list | None = None, gate: list | None = None):
            pg = b.new_page(viewport={"width": 1280, "height": 900})
            if hold:
                pg.route("**/dashboard_detail.js*", lambda route: None)   # never resolves
            if gate is not None:
                # Held, but releasable: the test opens a sheet, checks the waiting state, then
                # fulfills the route from the real file and watches the panels fill in. That is
                # the only way to see the loader's redraw, which is what lets the shard carry
                # numbers at all (2026-11-08, 13:00).
                pg.route("**/dashboard_detail.js*", lambda route: gate.append(route))
            if watch is not None:      # registered before navigation, or the first requests are missed
                pg.on("request", lambda r: watch.append(r.url.split("?")[0].rsplit("/", 1)[-1]))
            pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
            pg.wait_for_function("typeof D !== 'undefined' && typeof openStockDetail === 'function'",
                                 timeout=60000)
            return pg

        yield make, ticker, detail["detail"][ticker]["about"], d / "dashboard_detail.js"
        b.close()


def _about_text(page) -> str:
    return re.sub(r"\s+", " ", page.evaluate(
        "document.getElementById('modal-about-text').textContent")).strip()


def test_opening_a_stock_fetches_the_description_and_shows_it(page_and_ticker):
    """The behaviour, end to end: the text is not in the first-paint payload, and it is on the
    screen a moment after the sheet opens."""
    make, ticker, about, _f = page_and_ticker
    page = make()
    assert page.evaluate(f"!('about' in (D.stock_detail['{ticker}'] || {{}}))"), (
        "the first-paint payload already had it - this test would pass without the loader")
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    page.wait_for_function("typeof DETAIL_LOADED !== 'undefined' && DETAIL_LOADED", timeout=30000)
    shown = _about_text(page)
    assert shown[:60] == re.sub(r"\s+", " ", about).strip()[:60], shown[:120]
    assert page.evaluate("getComputedStyle(document.getElementById('modal-about')).display") != "none"
    page.close()


def test_it_is_the_last_of_the_three_payloads_to_be_requested(page_and_ticker):
    """What the split actually buys, stated as the measurement rather than as an intention.

    The page is usable before this file is asked for: the ranking payload goes first, the
    context layer second, and the sheet-only prose third - and the third request comes from the
    sheet machinery (on an untouched page, the idle pre-warm), not from the start-up block.
    Asserted as an **order**, which is true on any connection, rather than as a timing.
    """
    make, _t, _a, _f = page_and_ticker
    seen: list[str] = []
    page = make(watch=seen)
    page.wait_for_function("typeof DETAIL_LOADED !== 'undefined' && DETAIL_LOADED", timeout=60000)
    js = [n for n in seen if n.endswith(".js")]
    assert "dashboard_detail.js" in js, js
    assert js.index("dashboard_detail.js") == len(js) - 1, js
    assert js.index("dashboard_data.js") < js.index("dashboard_context.js") < js.index("dashboard_detail.js"), js
    page.close()


def test_while_it_is_in_flight_the_block_says_so(page_and_ticker):
    """The held-route case, the pattern `tests/test_case_against.py` uses for the context layer.
    A reader on a slow connection must not be told the company has no description."""
    make, ticker, _a, _f = page_and_ticker
    page = make(hold=True)
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    assert "Loading the company description" in _about_text(page)
    assert page.evaluate("DETAIL_PENDING === true && DETAIL_LOADED === false")
    page.close()


# ------------------------------------------------- the file carries numbers (2026-11-08, 13:00)
def _peer_cells(page):
    """The subject's row in the peer table, cell by cell. Positions 2..9 are the eight figure
    columns: market cap, P/E, revenue growth, net margin, ROIC-or-ROE, D/E, dividend yield,
    FCF yield. The first, fourth and seventh of those come from ``financials``."""
    return page.evaluate("""() => { const r = document.querySelector('#modal-peers tr.peer-row-self');
        return r ? [...r.querySelectorAll('td')].map(e => e.textContent.trim()) : null; }""")


def test_while_it_is_in_flight_the_snapshot_says_so(page_and_ticker):
    """A wall of figures with nothing in it reads as a company that files nothing, which is true
    of none of them - so the Company snapshot distinguishes "not here yet" from "never
    published". The second sentence is the one the panel showed before `financials` moved, and
    it is still correct when the file has landed without the block
    (`tests/test_sheet_round_three.py::test_empty_sections_say_so`)."""
    make, ticker, _a, _f = page_and_ticker
    page = make(hold=True)
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    snap = page.evaluate("document.getElementById('modal-snapshot').textContent")
    assert "Loading the financial figures" in snap, snap[:200]
    assert "No financial figures were published" not in snap, snap[:200]
    page.close()


def test_while_it_is_in_flight_the_peer_cells_say_so_and_the_others_are_already_there(page_and_ticker):
    """The mixed case, which is the whole argument for letting the shard carry numbers: six of
    the eight figure columns come from ``raw`` and are on the screen at once; the three fed by
    ``financials`` (two of the eight columns plus the market cap) wait, and say so with an
    ellipsis rather than the em-dash that means "not published in this run"."""
    make, ticker, _a, _f = page_and_ticker
    page = make(hold=True)
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#modal-peers .peer-table", state="attached")
    cells = _peer_cells(page)
    assert cells, "no subject row in the peer table"
    figures = cells[2:10]
    assert figures.count("…") == 3, figures          # mcap, net margin, dividend yield
    assert "—" not in figures[:1], figures           # never an em-dash while waiting
    assert sum(1 for c in figures if c not in ("…", "—")) >= 4, figures
    page.close()


def test_when_it_lands_the_panels_that_read_it_are_redrawn(page_and_ticker):
    """The mechanism itself. Before 2026-11-08 the loader redrew only ``renderAbout``, so a field
    the Company snapshot or the peer table reads would have stayed in its waiting state until the
    reader closed and reopened the sheet. The route is held until the sheet is open and waiting,
    then fulfilled from the real file."""
    make, ticker, _a, path = page_and_ticker
    gate: list = []
    page = make(gate=gate)
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#modal-peers .peer-table", state="attached")
    assert "Loading the financial figures" in page.evaluate(
        "document.getElementById('modal-snapshot').textContent")
    assert _peer_cells(page)[2] == "…"

    page.wait_for_function("() => true")                  # let the route handler run
    assert gate, "the page never requested the detail file"
    gate[0].fulfill(path=str(path), content_type="application/javascript")

    page.wait_for_function("typeof DETAIL_LOADED !== 'undefined' && DETAIL_LOADED", timeout=30000)
    page.wait_for_function("""() => { const e = document.getElementById('modal-snapshot');
        return e && !/Loading the financial figures/.test(e.textContent); }""", timeout=30000)
    snap = page.evaluate("document.getElementById('modal-snapshot').textContent")
    assert "Market Cap" in snap, snap[:200]
    page.wait_for_function("""() => { const r = document.querySelector('#modal-peers tr.peer-row-self');
        return r && !/\u2026/.test(r.textContent); }""", timeout=30000)
    cells = _peer_cells(page)
    assert "…" not in cells[2:10], cells
    assert cells[2].startswith("$"), cells                # the market cap is a figure again
    page.close()
