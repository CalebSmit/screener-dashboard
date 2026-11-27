"""`pct` - each metric's sector percentile - moves to the sheet-only shard.

2026-11-26 (13:00), the next slice of `plan/universe-expansion.md` phase 1's `stock_detail`
shard. Measured on the published payload that morning
(`research/measurements/2026-11-08-stock-detail-field-weights.py`): `pct` 127.4 KB gzipped,
**15.1% of first paint**, the second-heaviest field after `raw`. Regenerated from the published
run: first paint 814,575 -> 693,214 bytes gzipped, folded payload 0 differences.

**The read-site audit corrected the plan's premise, as `inp`'s did.** The plan listed `raw` and
`pct` as read by "the movers strip and the compare drawer"; neither reads either (the drawer reads
`cat_scores`, the strip the rank history). Every reader of `pct` in the page is the open sheet:

- `unscoredWhy` (the unscored score tile's reason, and the workings' unscored note),
- `weightProfile` / `categoryWorkings` / `renderCategoryDetails` (the category workings),
- `metricDetailHtml` (a metric row's rank line),
- `workingsCsv` (the download).

The membership rule (`tests/test_detail_payload_split.py`): every surface that reads it is the
open sheet, **and every panel that prints it can say honestly that it is waiting.** So: the
workings say "Loading the sector percentiles…" (never "no data", which is a different and false
statement), an unscored tile says "Not scored: no data" without naming metrics it cannot yet
see, the loader redraws both keeping the categories the reader had open, and the CSV waits.

Outside the page (the 2026-11-09 lesson): every reader of the published payload folds the shard
back - `check_run_health.fold_shards` for the health gate and `audit_stock.py`; four test modules
gained the fold in this change (`test_calculation_reproducibility`, `test_context_layer`,
`test_sheet_round_three`, and `test_health_reads_every_shard`'s fixture). The build's own uses
(`calc_trace`, `stock_summary`, `ranking_audit`) run before `split_detail`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_dashboard as gd  # noqa: E402
import check_run_health as health  # noqa: E402
from tests import sheet_warm  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

PAYLOAD = ROOT / "dashboard_data.js"

# The audited readers. Adding a function here is a decision that it runs only in the open sheet,
# or says it is waiting while the file is in flight and is redrawn when it lands.
READERS = {"unscoredWhy", "weightProfile", "categoryWorkings", "renderCategoryDetails",
           "metricDetailHtml", "workingsCsv",
           # `y.pct` there is a valuation-history year's own field, from the context file
           "vhRow"}


# ------------------------------------------------------------------------------ 1. the audit
def test_pct_is_in_the_shard_and_raw_and_summary_are_not():
    assert "pct" in gd.DETAIL_KEYS
    # each has a first-paint reader outside the sheet (My Holdings). `raw` followed on 2026-11-27
    # with its one such figure kept behind as `vol` (tests/test_raw_in_detail_shard.py).
    for k in ("summary", "cat_scores", "contrib", "flags", "wp"):
        assert k not in gd.DETAIL_KEYS, k


def _enclosing_function(lines: list[str], i: int) -> str | None:
    for j in range(i, -1, -1):
        m = re.search(r"\bfunction\s+(\w+)\s*\(", lines[j])
        if m:
            return m.group(1)
    return None


@pytest.fixture(scope="module")
def page_src() -> str:
    """The page the generator builds, not the committed file (rule 10)."""
    return gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")


def test_every_reader_in_the_page_is_audited(page_src):
    lines = page_src.split("\n")
    found: dict = {}
    for i, line in enumerate(lines):
        if line.strip().startswith("//"):
            continue
        if re.search(r"\.pct\b|\[['\"]pct['\"]\]", line):
            found.setdefault(_enclosing_function(lines, i), []).append(line.strip()[:100])
    assert found, "found no reader of .pct at all - the scan is broken, not the page clean"
    outside = {f: v for f, v in found.items() if f not in READERS}
    assert not outside, ("a function outside the audited readers reads `pct`, which is not in "
                         f"first paint since 2026-11-26: {outside}")


def test_each_reader_tolerates_a_stock_without_percentiles(page_src):
    """After the file lands a stock with no percentile at all carries no `pct` key (the split
    drops an all-null block); `s.pct[m]` would throw there, so every read goes through a default."""
    for fn in ("weightProfile", "unscoredWhy", "categoryWorkings", "metricDetailHtml"):
        start = page_src.index(f"function {fn}(")
        nxt = page_src.find("\n    function ", start + 10)
        body = page_src[start:nxt if nxt > 0 else start + 20000]
        assert not re.search(r"\bs\.pct\[", body), fn


def test_the_loader_redraws_the_workings_and_the_csv_waits(page_src):
    loader = page_src[page_src.index("function loadDetail("):][:4000]
    assert "redrawPercentileSurfaces()" in loader
    assert loader.index("redrawPercentileSurfaces()") < loader.index("redrawOpenMetricDetails()")
    dl = page_src[page_src.index("function downloadWorkings("):][:1500]
    assert "detailPending(s0, 'pct')" in dl and "ensureDetail()" in dl


# ------------------------------------------------------------------ 2. the page, re-split
@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """The published payload folded whole, then split by the generator's own rule."""
    if sync_api is None:
        pytest.skip("playwright not installed")
    if not PAYLOAD.exists():
        pytest.skip("MISSING DATA: no published dashboard_data.js")
    text = PAYLOAD.read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"):].rstrip().rstrip(";"))
    data = health.fold_shards(data, health._dir_reader(ROOT))
    if not (data.get("weights") or {}).get("profiles"):
        pytest.skip("MISSING DATA: payload publishes no weight tables, so no workings are drawn")
    # Premise, asserted rather than assumed: the first stock by rank with a scored Valuation
    # category and at least one Valuation percentile. Never a named stock.
    ranked = sorted((r for r in data["table_data"] if r.get("Rank")), key=lambda r: (r["Rank"], r["Ticker"]))
    ticker = None
    for r in ranked:
        s = data["stock_detail"].get(r["Ticker"]) or {}
        if (s.get("cat_scores") or {}).get("valuation") is not None and any(
                v is not None for v in (s.get("pct") or {}).values()):
            ticker = r["Ticker"]
            break
    if ticker is None:
        pytest.skip("MISSING DATA: no ranked stock with a scored Valuation and percentiles")
    for s in data["stock_detail"].values():
        s.pop("ctx", None)
    for k in gd.CONTEXT_KEYS:
        data.pop(k, None)
    detail = gd.split_detail(data)
    assert not any("pct" in s for s in data["stock_detail"].values())
    assert any("pct" in v for v in detail["detail"].values())
    d = tmp_path_factory.mktemp("pct_shard")
    det_text = "window.SCREENER_DETAIL = " + json.dumps(detail) + ";"
    (d / "dashboard_detail.js").write_text(det_text, encoding="utf-8")
    (d / "dashboard_data.js").write_text("window.SCREENER_DATA = " + json.dumps(data) + ";", encoding="utf-8")
    ctx = ROOT / "dashboard_context.js"
    if ctx.exists():
        (d / "dashboard_context.js").write_text(ctx.read_text(encoding="utf-8"), encoding="utf-8")
    version = gd.hashlib.md5(det_text.encode("utf-8")).hexdigest()[:12]
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v", detail_version=version),
                                  encoding="utf-8")
    with sync_api.sync_playwright() as pw:
        try:
            b = pw.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        errors: list[str] = []

        def make(gate: list, fail: bool = False):
            pg = b.new_page(viewport={"width": 1280, "height": 900})
            pg.on("pageerror", lambda e: errors.append(str(e)))
            if fail:
                pg.route("**/dashboard_detail.js*", lambda route: (gate.append(1), route.abort()))
            else:
                pg.route("**/dashboard_detail.js*", lambda route: gate.append(route))   # held
            pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
            pg.wait_for_function("typeof D !== 'undefined' && typeof openStockDetail === 'function'",
                                 timeout=60000)
            sheet_warm.settle(pg)
            return pg

        yield make, ticker, d / "dashboard_detail.js"
        b.close()
        assert not errors, errors


def _release(page, gate, path):
    page.wait_for_function("() => true")
    assert gate, "the page never requested the detail file"
    gate[0].fulfill(path=str(path), content_type="application/javascript")
    page.wait_for_function("document.documentElement.dataset.detail === 'loaded'", timeout=30000)


def _section(page, cat: str) -> str:
    """The section's text whether or not it is open (a collapsed body is hidden from inner_text)."""
    return re.sub(r"\s+", " ", page.text_content(f"#cat-detail-{cat}") or "")


def test_the_workings_say_they_are_loading_then_draw_and_keep_what_was_open(site):
    """Before the file: every category says it is loading and none says "no data". After: the
    table is drawn, its points add up to the published score, and the category the reader opened
    is still open."""
    make, ticker, path = site
    gate: list = []
    page = make(gate)
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    page.evaluate("openWorkings('valuation')")
    for cat in ("valuation", "quality", "momentum"):
        t = _section(page, cat)
        assert "Loading the sector percentiles" in t, t[:300]
        assert "no data" not in t.lower()
    assert page.locator("#modal-categories tr.metric-row").count() == 0
    assert page.locator("#modal-categories .wk-total").count() == 0     # no total without its rows
    # what first paint has is already there: the category's score and its points
    assert "Score" in _section(page, "valuation") and "pts" in _section(page, "valuation")

    _release(page, gate, path)
    page.wait_for_selector("#cat-detail-valuation tr.metric-row", timeout=30000)
    t = _section(page, "valuation")
    assert "Loading the sector percentiles" not in t
    open_now = page.evaluate("""() => [...document.querySelectorAll('#modal-categories .cat-detail-section')]
        .filter(x => !x.classList.contains('collapsed')).map(x => x.dataset.cat)""")
    assert open_now == ["valuation"], open_now
    assert page.get_attribute("#cat-detail-valuation .cat-detail-header", "aria-expanded") == "true"
    total = page.inner_text("#cat-detail-valuation .wk-total .wk-points")
    assert "!" not in total, f"the redrawn workings do not add up: {total}"
    page.close()


def test_an_unscored_tile_names_no_metric_until_the_percentiles_land(site):
    """Which metrics were empty is in the shard; until it lands the tile may only say what is true
    of every unscored category. A stand-in stock, so the premise does not depend on the day's data."""
    make, _ticker, path = site
    gate: list = []
    page = make(gate)
    js = """(withPct) => {
        const prof = (D.weights.profiles || {}).valuation || {};
        const table = prof.generic || {};
        const weighted = Object.keys(table).filter(m => table[m] > 0);
        const s = { cat_scores: { valuation: null }, contrib: { valuation: 0 }, flags: {}, wp: {} };
        if (withPct) { s.pct = {}; weighted.forEach(m => s.pct[m] = null); }
        return { n: weighted.length, w: unscoredWhy('valuation', s) };
    }"""
    r = page.evaluate(js, False)
    assert r["w"]["short"] == "Not scored: no data", r
    assert r["w"]["metrics"] == []
    # present always wins: a stock carrying its percentiles is told its reason in full, as before
    r = page.evaluate(js, True)
    assert r["w"]["metrics"] and len(r["w"]["metrics"]) == r["n"], r
    assert r["w"]["short"] != "Not scored: no data" or r["n"] == 0
    page.close()


def test_the_workings_download_waits_for_the_percentiles(site):
    """A file written before the shard landed would print "no data" on every metric row."""
    make, ticker, path = site
    gate: list = []
    page = make(gate)
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    page.evaluate("""() => { window.__csv = []; const orig = URL.createObjectURL;
        URL.createObjectURL = b => { window.__csv.push(b); return orig.call(URL, b); }; }""")
    page.evaluate("downloadWorkings()")
    assert page.evaluate("window.__csv.length") == 0, "it wrote the file before the percentiles arrived"
    _release(page, gate, path)
    page.wait_for_function("window.__csv.length === 1", timeout=30000)
    text = page.evaluate("window.__csv[0].text()")
    rows = [ln for ln in text.splitlines() if ln.startswith("Valuation,")]
    assert rows and any(",no data," not in ln for ln in rows), rows[:5]
    page.close()


def test_a_failed_load_says_so_rather_than_no_data_and_does_not_loop(site):
    """And it asks again only when a sheet opens. Until 2026-11-26 the failure's own redraw asked
    again at once (renderAbout -> loadDetail), so a refused file was re-requested in a tight loop
    for as long as the sheet stayed open - hundreds of requests a second."""
    make, ticker, _path = site
    asked: list = []
    page = make(asked, fail=True)
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    page.wait_for_function("() => !DETAIL_PENDING && DETAIL_FAILED", timeout=30000)
    page.wait_for_timeout(1500)
    n = len(asked)
    assert n <= 2, f"the page asked for a refused file {n} times with one sheet open"
    t = _section(page, "valuation")
    assert "could not be loaded" in t, t[:300]
    assert "no data" not in t.lower()
    # the next sheet that opens tries again, once
    page.evaluate("closeModal()")
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_function("() => !DETAIL_PENDING && DETAIL_FAILED", timeout=30000)
    page.wait_for_timeout(500)
    assert n < len(asked) <= n + 1, (n, len(asked))
    page.close()
