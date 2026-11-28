"""`eq4` and `roe5` - the quarters and fiscal years behind four metrics' workings - move to the
sheet-only shard.

2026-11-22 (13:00), the next slice of `plan/universe-expansion.md` phase 1's `stock_detail`
shard, after `inp` that morning. Measured on the published payload
(`research/measurements/2026-11-08-stock-detail-field-weights.py`): `roe5` 65.0 KB and `eq4`
48.8 KB gzipped, **12.1% of first paint together** - the two heaviest fields left after `raw`,
`pct` and `summary`, which cannot move whole (their readers are live at first paint).

The membership rule (`tests/test_detail_payload_split.py`): a field may move when every surface
that reads it is the open sheet, **or can say honestly that it is waiting**.

**The read-site audit, done before the move.** In the page:

- `roe5` is read by `metricDetailHtml` alone (earnings variability's five-year ROE table).
- `eq4` is read by `metricDetailHtml` (the three surprise metrics' quarter table) **and by
  `renderReporting`** - the "Reporting soon" calendar's *Last quarter* column. That section is
  collapsed at first paint and is not the sheet, so it is the case the rule's second half is
  for: it prints "…" while the file is in flight (the options column beside it already does the
  same for the context file), asks for the file when it is drawn, and the loader redraws it.

Outside the page (the 2026-11-09 lesson): nothing in `scripts/` or any scoring module reads
either field; `check_run_health.fold_shards` folds every shard generically; the four test
modules that check the published workings (`test_metric_lineage`, `test_methodology_claims_audit`,
`test_dashboard_printed_figures`, `test_sec_fundamentals`) all fold the shard back - the last one
gained its fold in this change, or it would have skipped as "payload predates earnings
variability".
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

# The audited readers, per field. Adding a function here is a decision that it runs only in the
# open sheet, or says it is waiting while the file is in flight and is redrawn when it lands.
READERS = {"eq4": {"metricDetailHtml", "renderReporting"},
           "roe5": {"metricDetailHtml"}}
SURPRISE = ("analyst_surprise", "earnings_acceleration", "consecutive_beat_streak")


# ------------------------------------------------------------------------------ 1. the audit
def test_eq4_and_roe5_are_in_the_shard():
    for k in READERS:
        assert k in gd.DETAIL_KEYS, k
    for k in ("pio", "bn"):   # `pct` joined 2026-11-26, `raw` 2026-11-27, `summary` 2026-11-28 (their own modules)
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


@pytest.mark.parametrize("field", sorted(READERS))
def test_every_reader_in_the_page_is_audited(page_src, field):
    lines = page_src.split("\n")
    found: dict = {}
    for i, line in enumerate(lines):
        if re.search(rf"\.{field}\b", line):
            found.setdefault(_enclosing_function(lines, i), []).append(line.strip()[:100])
    assert found, f"found no reader of .{field} at all - the scan is broken, not the page clean"
    outside = {f: v for f, v in found.items() if f not in READERS[field]}
    assert not outside, (f"a function outside the audited readers reads `{field}`, which is not in "
                         f"first paint since 2026-11-22: {outside}")


def test_each_reader_checks_whether_the_file_is_in_flight(page_src):
    body = page_src[page_src.index("function metricDetailHtml("):][:20000]
    assert "detailPending(s, 'eq4')" in body and "detailPending(s, 'roe5')" in body
    rep = page_src[page_src.index("function renderReporting("):][:9000]
    assert "detailPending(x.s, 'eq4')" in rep
    assert "loadDetail(DETAIL_SRC)" in rep          # it asks for the file, it does not wait on luck
    loader = page_src[page_src.index("function loadDetail("):][:3000]
    assert "renderReporting()" in loader            # and the loader redraws it


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
    # Premise, asserted rather than assumed: the first stock by rank with published quarters and
    # five ROE years behind scored values. Never a named stock.
    ranked = sorted((r for r in data["table_data"] if r.get("Rank")), key=lambda r: (r["Rank"], r["Ticker"]))
    ticker = None
    for r in ranked:
        s = data["stock_detail"].get(r["Ticker"]) or {}
        raw = s.get("raw") or {}
        if (s.get("eq4") and raw.get("analyst_surprise") is not None
                and s.get("roe5") and raw.get("earnings_variability") is not None):
            ticker = r["Ticker"]
            break
    if ticker is None:
        pytest.skip("MISSING DATA: no ranked stock with published quarters and ROE years")
    for s in data["stock_detail"].values():
        s.pop("ctx", None)
    for k in gd.CONTEXT_KEYS:
        data.pop(k, None)
    detail = gd.split_detail(data)
    for k in READERS:
        assert not any(k in s for s in data["stock_detail"].values()), k
        assert any(k in v for v in detail["detail"].values()), k
    d = tmp_path_factory.mktemp("eq4_shard")
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

        def make(gate: list):
            pg = b.new_page(viewport={"width": 1280, "height": 900})
            pg.on("pageerror", lambda e: errors.append(str(e)))
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


def _txt(h: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))


def test_the_workings_say_they_are_waiting_then_show_the_quarters_and_years(site):
    """A surprise row opened before the file lands must not read as "no quarters were published",
    and earnings variability must not read "Needs all five years"; both fill in once it lands."""
    make, ticker, path = site
    gate: list = []
    page = make(gate)
    js = f"(m) => metricDetailHtml(m, D.stock_detail['{ticker}'], null)"
    for m in SURPRISE:
        h = _txt(page.evaluate(js, m))
        assert "Loading the quarterly figures" in h, h[:300]
        assert "Actual EPS" not in h
    h = _txt(page.evaluate(js, "earnings_variability"))
    assert "Loading the five years of filings" in h, h[:300]
    assert "Needs all five years" not in h

    # Since 2026-11-26 the percentiles share the file, so no metric row is drawn before it lands
    # (the category says it is loading); once it has, the row a reader opens has its quarters.
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    page.evaluate("openWorkings('revisions')")
    assert page.locator("#cat-detail-revisions tr[data-metric='analyst_surprise']").count() == 0
    row = lambda: re.sub(r"\s+", " ", page.inner_text("#cat-detail-revisions tr.wk-detail"))  # noqa: E731

    _release(page, gate, path)
    page.click("#cat-detail-revisions tr[data-metric='analyst_surprise'] .wk-info")
    assert "Loading the quarterly figures" not in row()
    filled = row().lower()                          # inner_text applies the headers' text-transform
    assert "actual eps" in filled and "median of the" in filled, filled[:300]
    h = _txt(page.evaluate(js, "earnings_variability"))
    assert "Sample standard deviation of the five ROEs" in h and "Loading" not in h
    page.close()


def test_the_reporting_calendar_waits_asks_and_redraws(site):
    """The one reader outside the sheet: the Last quarter column says "…", the section asks for
    the file itself, and the loader fills the column."""
    make, _ticker, path = site
    gate: list = []
    page = make(gate)
    # Premise, asserted: a stock reporting within a window the panel can show that has quarters.
    # The window is widened by the panel's own setter, so the premise does not depend on today.
    page.evaluate("() => { RP.days = 400; RP.scope = 'all'; }")
    n = page.evaluate("() => reportingRows(400, 'all').length")
    if not n:
        pytest.skip("MISSING DATA: no scored stock with a coming report date in the payload")
    page.wait_for_function("() => true")             # let any idle pre-warm request reach the route
    asked_before = len(gate)
    page.evaluate("() => { RP_DONE = true; renderReporting(); }")
    cells = page.evaluate("""() => [...document.querySelectorAll('#reporting-body tr.rp-row')]
        .map(tr => tr.children[4].textContent.trim())""")
    assert cells and all(c == "…" for c in cells), cells[:10]
    # a phone reads the same figure under the company name: it waits there too, not vanishes
    mob = page.evaluate("""() => [...document.querySelectorAll('#reporting-body tr.rp-row .rp-m')]
        .map(e => e.textContent)""")
    assert len(mob) == len(cells) and all("Last quarter vs estimate …" in m for m in mob), mob[:5]
    page.wait_for_function("() => true")
    # Either the pre-warm had already asked (the request is held, so it is still in flight) or the
    # calendar asked itself; a waiting column with no request in flight would wait forever.
    assert asked_before or gate, "the calendar drew a waiting column without asking for the file"
    if not asked_before:
        assert page.evaluate("DETAIL_PENDING")
    _release(page, gate, path)
    page.wait_for_function("""() => [...document.querySelectorAll('#reporting-body tr.rp-row')]
        .some(tr => /%/.test(tr.children[4].textContent))""", timeout=30000)
    cells = page.evaluate("""() => [...document.querySelectorAll('#reporting-body tr.rp-row')]
        .map(tr => tr.children[4].textContent.trim())""")
    assert "…" not in cells
    page.close()
