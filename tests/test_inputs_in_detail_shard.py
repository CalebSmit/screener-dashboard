"""`inp` - the reported figures behind every metric's equation - moves to the sheet-only shard.

2026-11-22, the next slice of `plan/universe-expansion.md` phase 1's `stock_detail` shard (the
plan's "Next in phase 1"). `inp` was **13.8%** of the 2026-11-08 first-paint payload
(`research/measurements/2026-11-08-stock-detail-field-weights.py`), the heaviest field left in it.

The membership rule the shard has carried since 2026-11-08 (`tests/test_detail_payload_split.py`):

    a field may move when every surface that reads it is the open sheet, and every panel that
    prints it can say honestly that it is waiting.

**The read-site audit, done before the move.** In the page `inp` is read in four functions, all
inside the open sheet: `metricDetailHtml` and `equationHtml` (a metric row, built when the reader
opens it), `renderProvenance` (the recomputed-EV pair) and `workingsCsv` (the sheet's download).
The 2026-11-08 comment said the calculation traces read it; they do not - the build's
`calc_trace` runs on the whole payload before `split_detail`, as do the input-mismatch and
capped-figure checks and `ranking_audit`. **Outside the page** (the 2026-11-09 lesson: grep for a
field outside the page too) the published payload's readers are `sec_crosscheck`, through
`check_run_health.fold_shards`, which folds every shard generically; `scripts/audit_stock.py`; and
three test modules, which now fold the shard back as the gate does.

`summary`, `raw` and `pct` stay in first paint: My Holdings prints six kinds of `summary`
sentence, and the movers strip and compare drawer read `raw`/`pct` - none of them the sheet.

The tests:

1. a tripwire on the audit - every `.inp` read in the page lies in a function named here, so a
   new reader outside the sheet fails until someone decides whether it can wait;
2. the waiting states and the redraw, on the published payload re-split by the generator's own
   `split_detail` (so they test what the next run publishes, whether or not the committed payload
   has been regenerated since this change);
3. the download waits for the inputs instead of writing a file without them.
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
import metric_lineage as ml  # noqa: E402
import check_run_health as health  # noqa: E402
from tests import sheet_warm  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

PAYLOAD = ROOT / "dashboard_data.js"

# The audited readers. A function added here is a decision that it, too, runs only in the open
# sheet and says it is waiting while `inp` is in flight.
SHEET_READERS = {"metricDetailHtml", "equationHtml", "renderProvenance", "workingsCsv",
                 "inputsPending", "downloadWorkings"}


# ------------------------------------------------------------------------------ 1. the audit
def test_inp_is_in_the_sheet_only_shard():
    assert "inp" in gd.DETAIL_KEYS
    # the small fields the category table itself reads stay in first paint
    for k in ("inp_bad", "xcap", "pio", "raw", "pct", "summary"):
        assert k not in gd.DETAIL_KEYS, k


def _enclosing_function(lines: list[str], i: int) -> str | None:
    for j in range(i, -1, -1):
        m = re.search(r"\bfunction\s+(\w+)\s*\(", lines[j])
        if m:
            return m.group(1)
    return None


def test_every_reader_of_inp_in_the_page_is_inside_the_sheet():
    """Scans the page the generator builds, not the committed file (rule 10)."""
    page = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    lines = page.split("\n")
    readers = {}
    for i, line in enumerate(lines):
        if re.search(r"\.inp\b(?!_)", line):
            readers.setdefault(_enclosing_function(lines, i), []).append(line.strip()[:100])
    assert readers, "found no reader of .inp at all - the scan is broken, not the page clean"
    outside = {f: v for f, v in readers.items() if f not in SHEET_READERS}
    assert not outside, ("a function outside the audited sheet readers reads `inp`, which is not "
                         f"in first paint since 2026-11-22: {outside}")


def test_the_waiting_states_are_in_every_audited_reader():
    """Each panel that prints `inp` checks whether it is still in flight."""
    page = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    for fn in ("metricDetailHtml", "equationHtml", "downloadWorkings"):
        body = page[page.index(f"function {fn}("):][:6000]
        assert "inputsPending(" in body, fn
    prov = page[page.index("function renderProvenance("):][:12000]
    assert "Loading the figure used here" in prov


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
    # Premise, asserted rather than assumed: the first stock by rank that is not a financial
    # and has an earnings-yield equation the page can fill. Never a named stock.
    ranked = sorted((r for r in data["table_data"] if r.get("Rank")), key=lambda r: (r["Rank"], r["Ticker"]))
    ticker = None
    for r in ranked:
        s = data["stock_detail"].get(r["Ticker"]) or {}
        if (s.get("sector") != "Financials" and (s.get("raw") or {}).get("earnings_yield") is not None
                and ml.choose_template("earnings_yield", s.get("inp") or {}) is not None):
            ticker = r["Ticker"]
            break
    if ticker is None:
        pytest.skip("MISSING DATA: no ranked non-financial stock with an earnings-yield equation")
    for k in ("ctx",):                       # the context layer is its own file; keep it out
        for s in data["stock_detail"].values():
            s.pop(k, None)
    for k in gd.CONTEXT_KEYS:
        data.pop(k, None)
    detail = gd.split_detail(data)
    assert not any("inp" in s for s in data["stock_detail"].values())
    assert any("inp" in v for v in detail["detail"].values())
    d = tmp_path_factory.mktemp("inp_shard")
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

        def make(gate: list | None = None):
            pg = b.new_page(viewport={"width": 1280, "height": 900})
            pg.on("pageerror", lambda e: errors.append(str(e)))
            if gate is not None:     # held until the test releases it
                pg.route("**/dashboard_detail.js*", lambda route: gate.append(route))
            pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
            pg.wait_for_function("typeof D !== 'undefined' && typeof openStockDetail === 'function'",
                                 timeout=60000)
            sheet_warm.settle(pg)
            return pg

        yield make, ticker, d / "dashboard_detail.js"
        b.close()
        assert not errors, errors


def _row_text(page) -> str:
    return re.sub(r"\s+", " ", page.inner_text("#cat-detail-valuation tr.wk-detail"))


def _release(page, gate, path):
    page.wait_for_function("() => true")
    assert gate, "the page never requested the detail file"
    gate[0].fulfill(path=str(path), content_type="application/javascript")
    page.wait_for_function("document.documentElement.dataset.detail === 'loaded'", timeout=30000)


def test_an_open_metric_row_says_it_is_waiting_then_fills_in(site):
    """The row a reader opened before the file landed must not read as "no inputs were
    published", and must not stay that way once they have."""
    make, ticker, path = site
    gate: list = []
    page = make(gate=gate)
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    page.evaluate("openWorkings('valuation')")
    page.click("#cat-detail-valuation tr[data-metric='earnings_yield'] .wk-info")
    waiting = _row_text(page)
    assert "Loading the reported figures" in waiting, waiting[:300]
    assert "Net income (TTM)" not in waiting           # no input row while there are no inputs
    assert "Ranked" in waiting and "percentile" in waiting   # what first paint has is already there

    _release(page, gate, path)
    page.wait_for_function("""() => { const r = document.querySelector('#cat-detail-valuation tr.wk-detail');
        return r && !/Loading the reported figures/.test(r.textContent); }""", timeout=30000)
    filled = _row_text(page)
    assert "Net income (TTM)" in filled and "Market cap" in filled, filled[:300]
    assert "=" in filled
    page.close()


def test_the_recomputed_enterprise_value_says_it_is_waiting(site):
    make, _ticker, _path = site
    gate: list = []
    page = make(gate=gate)
    txt = page.evaluate("""() => { renderProvenance({ metric_count: 25, metric_total: 28,
            data_source: 'quarterly', sector: 'Industrials', asof: {},
            flags: { ev_flag: true, ev_api: 120e9 } });
        return document.getElementById('modal-provenance').textContent; }""")
    assert "Enterprise value recomputed" in txt
    assert "Loading the figure used here" in txt and "undefined" not in txt
    # with the inputs present it states both figures, as before the move
    txt = page.evaluate("""() => { renderProvenance({ metric_count: 25, metric_total: 28,
            data_source: 'quarterly', sector: 'Industrials', asof: {}, inp: { ev_used: 80e9 },
            flags: { ev_flag: true, ev_api: 120e9 } });
        return document.getElementById('modal-provenance').textContent; }""")
    assert "$120.0B" in txt and "$80.0B" in txt and "Loading" not in txt
    page.close()


def test_the_workings_download_waits_for_the_inputs(site):
    """A file written before the shard landed would silently lack its INPUTS block."""
    make, ticker, path = site
    gate: list = []
    page = make(gate=gate)
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    page.evaluate("""() => { window.__csv = []; const orig = URL.createObjectURL;
        URL.createObjectURL = b => { window.__csv.push(b); return orig.call(URL, b); }; }""")
    page.evaluate("downloadWorkings()")
    assert page.evaluate("window.__csv.length") == 0, "it wrote the file before the inputs arrived"
    _release(page, gate, path)
    page.wait_for_function("window.__csv.length === 1", timeout=30000)
    text = page.evaluate("window.__csv[0].text()")
    assert "INPUTS: as reported" in text
    page.close()
