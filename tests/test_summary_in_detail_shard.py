"""`summary` - the "Why it ranks here" sentences - moves to the sheet-only shard; the six kinds My
Holdings prints stay in first paint as `hsum`.

2026-11-28 (weekend 06:00), the next slice of `plan/universe-expansion.md` phase 1. Measured on the
published payload first (`logs/summary_weights.py`): `summary` was 69.6 KB of the 495 KB gzipped
first paint, of which the holdings' six kinds are 16.2 KB. Regenerated from the published run: first
paint **494,961 -> 426,815 bytes gzipped (-13.8%)**, folded payload **0 differences** apart from the
new `hsum` copies, each equal to its stock's summary filtered to `HOLDINGS_FACTS` (499 of 499).

**The read-site audit.** In the page `summary` is read by three functions:

- `renderSummary` - the sheet's "Why it ranks here", every kind. Says "Loading why it ranks here…"
  until the file lands, and is redrawn by the loader.
- `caseItems` - the sheet's "The case against": five of the holdings' kinds (flags, confidence,
  change, input_churn, earnings) and one more, the analyst-target sentence. It reads first paint's
  copy until the file lands, and only the target line waits ("Loading the analyst-target line…").
- My Holdings' card (`holdingCard`) - **the one reader outside the sheet**, live at first paint for a
  reader with saved names, printing the six `HOLDINGS_FACTS` kinds. It reads `hsum`.

All three go through `factsOf`, which prefers the whole list and falls back to `hsum`, so a payload
built by hand with `summary` inline renders exactly as before. The page's `HOLDINGS_FACTS` is written
from `generate_dashboard.HOLDINGS_FACTS`, the tuple `split_detail` filters by: one list, not two.
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

# The audited readers of a stock's `summary`. Adding a function here is a decision that it runs only
# in the open sheet and says it is waiting while the file is in flight, or reads only the kinds
# first paint keeps.
READERS = {"factsOf", "summaryPending", "renderSummary"}


# ------------------------------------------------------------------------------ 1. the audit
def test_summary_is_in_the_shard_and_the_holdings_kinds_stay_behind():
    assert "summary" in gd.DETAIL_KEYS
    assert "hsum" not in gd.DETAIL_KEYS
    assert "hsum" in gd.FIRST_PAINT_COPIES
    assert gd.HOLDINGS_FACTS == ("change", "change_driver", "input_churn", "flags", "confidence", "earnings")


def test_split_detail_moves_summary_and_keeps_the_holdings_kinds_in_order():
    facts = [{"k": "rank", "t": "r"}, {"k": "flags", "t": "f"}, {"k": "drivers", "t": "d"},
             {"k": "change", "t": "c"}, {"k": "target", "t": "t"}, {"k": "earnings", "t": "e"}]
    data = {"stock_detail": {
        "AAA": {"rank": 1, "summary": list(facts)},
        "BBB": {"rank": 2, "summary": [{"k": "rank", "t": "r"}]},   # none of the six: no hsum key
        "CCC": {"rank": 3},                                           # no summary at all
    }}
    out = gd.split_detail(data)
    sd = data["stock_detail"]
    assert not any("summary" in s for s in sd.values())
    assert sd["AAA"]["hsum"] == [facts[1], facts[3], facts[5]]       # the summary's order, not the tuple's
    assert "hsum" not in sd["BBB"] and "hsum" not in sd["CCC"]
    assert out["detail"]["AAA"]["summary"] == facts
    assert out["detail"]["BBB"]["summary"] == [{"k": "rank", "t": "r"}]


@pytest.fixture(scope="module")
def page_src() -> str:
    """The page the generator builds, not the committed file (rule 10)."""
    return gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")


def test_the_page_filters_by_the_same_tuple(page_src):
    m = re.search(r"const HOLDINGS_FACTS = (\[[^\]]*\]);", page_src)
    assert m, "the page has no HOLDINGS_FACTS"
    assert tuple(json.loads(m.group(1))) == gd.HOLDINGS_FACTS


def _enclosing_function(lines: list[str], i: int) -> str | None:
    for j in range(i, -1, -1):
        m = re.search(r"\bfunction\s+(\w+)\s*\(", lines[j])
        if m:
            return m.group(1)
    return None


def test_every_reader_in_the_page_is_audited(page_src):
    lines = page_src.split("\n")
    found: dict = {}
    for i, line in enumerate(lines):
        if line.strip().startswith("//"):
            continue
        # `.summary` as a field, never a CSS class (`.summary-fact`) or the <summary> element
        if re.search(r"\.summary(?![\w-])|\[['\"]summary['\"]\]", line):
            found.setdefault(_enclosing_function(lines, i), []).append(line.strip()[:100])
    assert found, "found no reader of .summary at all - the scan is broken, not the page clean"
    outside = {f: v for f, v in found.items() if f not in READERS}
    assert not outside, ("a function outside the audited readers reads `summary`, which is not in "
                         f"first paint since 2026-11-28: {outside}")


def test_the_holdings_card_and_the_case_against_read_through_factsOf(page_src):
    card = page_src[page_src.index("function holdingCard("):][:4000]
    assert "factsOf(s)" in card
    case = page_src[page_src.index("function caseItems("):][:1200]
    assert "factsOf(s)" in case and "summaryPending(s)" in case


@pytest.mark.skipif(not PAYLOAD.exists(), reason="MISSING DATA: no published payload")
def test_the_published_first_paint_carries_hsum_and_not_summary():
    text = PAYLOAD.read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"):].rstrip().rstrip(";"))
    sd = data["stock_detail"]
    if any("summary" in s for s in sd.values()):
        pytest.skip("MISSING DATA: the published payload predates the `summary` shard")
    folded = health.fold_shards(json.loads(text[text.index("{"):].rstrip().rstrip(";")),
                                health._dir_reader(ROOT))
    with_summary = [t for t, s in folded["stock_detail"].items() if s.get("summary")]
    assert with_summary, "no stock has a summary once folded - the shard did not fold"
    for t in with_summary:
        want = [f for f in folded["stock_detail"][t]["summary"] if f.get("k") in gd.HOLDINGS_FACTS]
        assert sd[t].get("hsum", []) == want, t


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
    # Premise, asserted rather than assumed: the best-ranked stock whose summary carries a target
    # sentence and whose mean target sits below its price (so The case against lists it), and the
    # best-ranked stock with at least one of the holdings' kinds. Never a named stock.
    ranked = sorted((r for r in data["table_data"] if r.get("Rank")), key=lambda r: (r["Rank"], r["Ticker"]))
    sd = data["stock_detail"]

    def kinds(t):
        return {f.get("k") for f in (sd.get(t) or {}).get("summary") or []}
    held = next((r["Ticker"] for r in ranked if kinds(r["Ticker"]) & set(gd.HOLDINGS_FACTS)), None)
    under = next((r["Ticker"] for r in ranked if "target" in kinds(r["Ticker"])
                  and isinstance(sd[r["Ticker"]].get("price"), (int, float))
                  and isinstance(sd[r["Ticker"]].get("pt_mean"), (int, float))
                  and 0 < sd[r["Ticker"]]["pt_mean"] < sd[r["Ticker"]]["price"]), None)
    if not held or not under:
        pytest.skip("MISSING DATA: no ranked stock with the holdings' kinds, or none below its target")
    want_notes = [f["t"] for f in sd[held]["summary"] if f["k"] in gd.HOLDINGS_FACTS]
    want_target = next(f["t"] for f in sd[under]["summary"] if f["k"] == "target")
    n_facts = len(sd[under]["summary"])
    for s in sd.values():
        s.pop("ctx", None)
    for k in gd.CONTEXT_KEYS:
        data.pop(k, None)
    detail = gd.split_detail(data)
    assert not any("summary" in s for s in data["stock_detail"].values())
    d = tmp_path_factory.mktemp("summary_shard")
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

        def make(gate: list, saved: list | None = None):
            pg = b.new_page(viewport={"width": 1280, "height": 900})
            pg.on("pageerror", lambda e: errors.append(str(e)))
            if saved is not None:
                pg.add_init_script("window.localStorage.setItem('screener_holdings_v1', "
                                   + json.dumps(json.dumps(saved)) + ");")
            pg.route("**/dashboard_detail.js*", lambda route: gate.append(route))   # held
            pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
            pg.wait_for_function("typeof D !== 'undefined' && typeof openStockDetail === 'function'",
                                 timeout=60000)
            sheet_warm.settle(pg)
            return pg

        yield make, held, want_notes, under, want_target, n_facts, d / "dashboard_detail.js", errors
        b.close()


def test_with_the_file_held_my_holdings_still_prints_its_notes(site):
    """The reader this slice had to keep: a saved name's review notes, at first paint, with the
    detail file not yet arrived. They are first paint's `hsum`, in the summary's order."""
    make, held, want_notes, _u, _t, _n, _f, errors = site
    gate: list = []
    page = make(gate, saved=[held])
    page.evaluate("typeof renderHoldings === 'function' && renderHoldings()")
    assert page.evaluate("DETAIL_LOADED") is False, "the detail file was meant to be held"
    notes = page.evaluate("[...document.querySelectorAll('.holding-card .holding-note')].map(e => e.textContent)")
    assert notes == want_notes, (notes[:3], want_notes[:3])
    assert not errors, errors
    page.close()


def test_with_the_file_held_the_sheet_waits_and_then_fills(site):
    """The sheet's two readers: Why it ranks here says it is loading and The case against waits for
    its target line only; then the file lands and both are drawn whole - no reopen needed."""
    make, _h, _w, under, want_target, n_facts, path, errors = site
    gate: list = []
    page = make(gate)
    page.evaluate(f"openStockDetail('{under}')")
    page.wait_for_selector("#modal-summary-body", state="attached")
    body = page.evaluate("document.getElementById('modal-summary-body').textContent")
    assert body == "Loading why it ranks here…", body
    case = page.evaluate("[...document.querySelectorAll('#modal-case-list .case-item')].map(e => e.dataset.k)")
    assert "target" not in case, case
    note = page.evaluate("""() => { const e = document.querySelector('#modal-case-list .case-note[data-k="wait"]');
        return e ? e.textContent : ''; }""")
    assert "Loading the analyst-target line…" in note, note
    assert gate, "the page never requested the detail file"
    gate[0].fulfill(path=str(path), content_type="application/javascript")
    page.wait_for_function("typeof DETAIL_LOADED !== 'undefined' && DETAIL_LOADED", timeout=30000)
    n = page.evaluate("document.querySelectorAll('#modal-summary-body .summary-fact').length")
    assert n == n_facts, (n, n_facts)
    got = page.evaluate("""() => { const e = document.querySelector('#modal-case-list .case-item[data-k="target"] .case-t');
        return e ? e.textContent : null; }""")
    assert got == want_target, got
    note = page.evaluate("""() => { const e = document.querySelector('#modal-case-list .case-note[data-k="wait"]');
        return e ? e.textContent : ''; }""")
    assert "analyst-target" not in note, note
    assert not errors, errors
    page.close()
