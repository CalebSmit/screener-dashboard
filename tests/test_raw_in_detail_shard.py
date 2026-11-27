"""`raw` - each metric's value as scored - moves to the sheet-only shard; volatility stays as `vol`.

2026-11-27 (13:00), the next slice of `plan/universe-expansion.md` phase 1. Re-measured on the
published payload first: after `pct` moved, `raw` was the heaviest field left in first paint.
Regenerated from the published run (`runs/c31edb9b89dc`, 2026-11-26 02:00): first paint
**693,214 -> 493,871 bytes gzipped (-28.8%)**, folded payload **0 differences**, and the new
`vol` equal to `raw.volatility` for all 497 stocks that have one.

**The read-site audit.** In the page `raw` is read by the open sheet - the peer table
(`buildPeerRow`, six of its nine columns), the workings and metric rows (`categoryWorkings`,
`peerContext`, `metricDetailHtml`, `equationHtml`), the workings CSV through the rows - and by two
surfaces that are **not** the sheet, both for one figure, volatility:

- My Holdings' concentration line (`holdingsConcentration`, live at first paint for a reader with
  saved names), and
- the context panel's "Realised, past year" row, drawn when `dashboard_context.js` lands, which
  can be before `dashboard_detail.js`.

That one figure stays in first paint as its own field, `vol` - copied from `raw` by
`split_detail` (`generate_dashboard.FIRST_PAINT_RAW`), never computed a second time - and both
read it through `volOf`, which falls back to `raw.volatility` so a payload built by hand renders
as before. The sheet's readers wait: the peer table draws with every figure "…" (the tickers and
ranks are first paint), and the workings already say "Loading the sector percentiles…" for
`pct`, which arrives in the same file.

Outside the page (the 2026-11-09 lesson - grep beyond the page): `write_brief.context_coverage`
counted earnings variability off `raw` and folds the shard now; four test modules and one
measurement script gained the fold, and one of them (`test_peer_row_one_path`) turned out to have
read `financials` as absent since it moved on 2026-11-08 and passed on `raw` alone.
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

# The audited readers of a stock's `raw`. Adding a function here is a decision that it runs only
# in the open sheet, or says it is waiting while the file is in flight and is redrawn when it lands.
READERS = {"buildPeerRow", "peerContext", "categoryWorkings", "metricDetailHtml", "equationHtml",
           # first paint's copy first, `raw` only for a payload built by hand
           "volOf",
           # `r.raw` there is a workings row's own field, built by categoryWorkings
           "renderCategoryDetails", "workingsCsv"}


# ------------------------------------------------------------------------------ 1. the audit
def test_raw_is_in_the_shard_and_volatility_stays_behind():
    assert "raw" in gd.DETAIL_KEYS
    assert gd.FIRST_PAINT_RAW == {"volatility": "vol"}
    # `vol` is a first-paint field, so it must not be a shard key itself
    assert "vol" not in gd.DETAIL_KEYS


def test_split_detail_moves_raw_and_keeps_vol_equal_to_it():
    data = {"stock_detail": {
        "AAA": {"rank": 1, "raw": {"volatility": 0.31, "roe": 0.2}},
        "BBB": {"rank": 2, "raw": {"volatility": None, "roe": 0.1}},   # a null is not carried
        "CCC": {"rank": 3},                                             # no raw at all
    }}
    out = gd.split_detail(data)
    sd = data["stock_detail"]
    assert not any("raw" in s for s in sd.values())
    assert sd["AAA"]["vol"] == 0.31
    assert "vol" not in sd["BBB"] and "vol" not in sd["CCC"]
    assert out["detail"]["AAA"]["raw"] == {"volatility": 0.31, "roe": 0.2}
    assert out["detail"]["BBB"]["raw"] == {"volatility": None, "roe": 0.1}


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
        if re.search(r"\.raw\b|\[['\"]raw['\"]\]", line):
            found.setdefault(_enclosing_function(lines, i), []).append(line.strip()[:100])
    assert found, "found no reader of .raw at all - the scan is broken, not the page clean"
    outside = {f: v for f, v in found.items() if f not in READERS}
    assert not outside, ("a function outside the audited readers reads `raw`, which is not in "
                         f"first paint since 2026-11-27: {outside}")


def test_the_two_readers_outside_the_sheet_read_first_paints_copy(page_src):
    conc = page_src[page_src.index("function holdingsConcentration("):page_src.index("function holdingCard(")]
    assert "volOf(r.s)" in conc and ".raw" not in conc
    row = page_src.index("<td>Realised, past year")
    ctx = page_src[row - 600:row]
    assert "volOf(s)" in ctx and "s.raw" not in ctx


def test_each_reader_tolerates_a_stock_without_raw(page_src):
    """After the file lands a stock with no figure at all carries no `raw` key (the split drops
    an all-null block); `s.raw[m]` would throw there."""
    for fn in ("peerContext", "equationHtml", "categoryWorkings"):
        start = page_src.index(f"function {fn}(")
        nxt = page_src.find("\n    function ", start + 10)
        body = page_src[start:nxt if nxt > 0 else start + 20000]
        assert not re.search(r"\b[so]\.raw\[", body), fn
    # metricDetailHtml keeps its registered sentence templates (claims.py) verbatim, so it gives
    # the stock an empty `raw` before the first of them instead
    md = page_src[page_src.index("function metricDetailHtml("):][:1200]
    assert "if (!s.raw) s = Object.assign({}, s, { raw: {} });" in md


@pytest.mark.skipif(not PAYLOAD.exists(), reason="MISSING DATA: no published payload")
def test_the_published_first_paint_carries_vol_and_not_raw():
    text = PAYLOAD.read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"):].rstrip().rstrip(";"))
    sd = data["stock_detail"]
    if any("raw" in s for s in sd.values()):
        pytest.skip("MISSING DATA: the published payload predates the `raw` shard")
    folded = health.fold_shards(json.loads(text[text.index("{"):].rstrip().rstrip(";")),
                                health._dir_reader(ROOT))
    with_vol = [t for t, s in folded["stock_detail"].items()
                if (s.get("raw") or {}).get("volatility") is not None]
    assert with_vol, "no stock has a volatility once folded - the shard did not fold"
    for t in with_vol:
        assert sd[t].get("vol") == folded["stock_detail"][t]["raw"]["volatility"], t


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
    # Premise, asserted rather than assumed: the two best-ranked stocks with a positive volatility
    # that differ. Never a named stock.
    ranked = sorted((r for r in data["table_data"] if r.get("Rank")), key=lambda r: (r["Rank"], r["Ticker"]))
    pair = []
    for r in ranked:
        v = ((data["stock_detail"].get(r["Ticker"]) or {}).get("raw") or {}).get("volatility")
        if isinstance(v, (int, float)) and v > 0 and all(v != p[1] for p in pair):
            pair.append((r["Ticker"], v))
        if len(pair) == 2:
            break
    if len(pair) < 2:
        pytest.skip("MISSING DATA: fewer than two ranked stocks with a volatility")
    for s in data["stock_detail"].values():
        s.pop("ctx", None)
    for k in gd.CONTEXT_KEYS:
        data.pop(k, None)
    detail = gd.split_detail(data)
    assert not any("raw" in s for s in data["stock_detail"].values())
    d = tmp_path_factory.mktemp("raw_shard")
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

        yield make, pair, d / "dashboard_detail.js", errors
        b.close()


def test_with_the_file_held_my_holdings_still_prints_the_risk_gap(site):
    """The reader this slice had to keep: a saved list's widest volatility gap, at first paint,
    with the detail file not yet arrived. It reads `vol`, which first paint carries."""
    make, pair, _f, errors = site
    gate: list = []
    page = make(gate, saved=[pair[0][0], pair[1][0]])
    page.evaluate("typeof renderHoldings === 'function' && renderHoldings()")
    txt = page.evaluate("""() => { const e = document.querySelector('.holdings-concentration');
        return e ? e.textContent : null; }""")
    assert page.evaluate("DETAIL_LOADED") is False, "the detail file was meant to be held"
    assert txt is not None, "no concentration block for a saved list of two"
    assert "Widest risk gap" in txt, txt[:300]
    hi, lo = sorted(pair, key=lambda p: -p[1])
    assert f"{hi[0]} at {hi[1] * 100:.0f}%" in txt, txt[:400]
    assert not errors, errors
    page.close()


def test_with_the_file_held_the_peer_table_waits_and_then_fills(site):
    """The sheet's reader: every figure waits with an ellipsis, then the file lands and the
    subject's row carries figures - no reopen needed."""
    make, pair, path, errors = site
    gate: list = []
    page = make(gate)
    ticker = pair[0][0]
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#modal-peers .peer-table", state="attached")
    row = page.evaluate("""() => { const r = document.querySelector('#modal-peers tr.peer-row-self');
        return r ? [...r.querySelectorAll('td')].map(e => e.textContent.trim()) : null; }""")
    assert row and row[2:10] == ["…"] * 8, row
    page.wait_for_function("() => true")
    assert gate, "the page never requested the detail file"
    gate[0].fulfill(path=str(path), content_type="application/javascript")
    page.wait_for_function("typeof DETAIL_LOADED !== 'undefined' && DETAIL_LOADED", timeout=30000)
    page.wait_for_function("""() => { const r = document.querySelector('#modal-peers tr.peer-row-self');
        return r && !/…/.test(r.textContent); }""", timeout=30000)
    row = page.evaluate("""() => [...document.querySelectorAll('#modal-peers tr.peer-row-self td')]
        .map(e => e.textContent.trim())""")
    assert any(c not in ("—", "…", "") for c in row[3:10]), row
    assert not errors, errors
    page.close()
