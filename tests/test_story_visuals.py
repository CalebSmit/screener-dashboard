"""Pictures that tell the story a section's sentence or table already states.

WHY THIS EXISTS - owner request, 2026-11-05: *"it needs to be displayed to tell a story so people
can understand."* A walk through the live page found the strongest facts held only in prose or in
a hidden table:

* **What changed** says which categories drove the reshuffle in a sentence ("Growth accounted for
  21%, Valuation 20% and Quality 17%"); the eight shares it is built from were published and never
  drawn. Now drawn, largest first, with the three fundamentals in the accent.
* **Track record** says "5 of 8" periods ahead of RSP; which ones, and by how much, sat in a closed
  table. Now one bar per period, green where the top 25 beat RSP.
* **Factor analytics** shades each column in ink only, so a sector that stands out had to be found
  by comparing numbers by eye. Now each factor's single highest and lowest sector is marked.
* The analyst-target bar cut its own "Low" label at the panel's edge ("ow $361.00").

Each picture must say what its words say, so the checks run **the generator's own page** on the
published payloads and assert relationships, never values: the bars are the published shares, the
green bars are the published count by the record's own rule, the marked cell is its column's
maximum. Nothing names a date, a level or a sector (CLAUDE.md rows "data->red" and "clock->red");
the price-target check opens its stock through ``tests/examples.py``.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
import run_overview  # noqa: E402
from tests import examples  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None


def _page_html() -> str:
    return gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")


def test_the_fundamentals_drawn_are_the_set_the_sentence_sums():
    html = _page_html()
    line = next(ln for ln in html.splitlines() if "const CHG_FUND = " in ln)
    assert json.loads(line.split("=", 1)[1].strip().rstrip(";")) == list(run_overview.FUNDAMENTAL)


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The generator's own page (never the committed file) with both published payloads."""
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("story")
    (d / "index.html").write_text(_page_html(), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof renderChangedShares === 'function' && typeof D !== 'undefined' && "
                             "window.SCREENER_CONTEXT !== undefined", timeout=60000)
        yield pg
        b.close()
        assert not errors, errors


# --------------------------------------------------------------------------- What moved the scores
_SHARES_JS = """(ov) => { renderChangedShares(ov); return [...document.querySelectorAll('#changed-shares .chs-row')].map(r => ({
    n: r.querySelector('.chs-n').textContent, v: r.querySelector('.chs-v').textContent,
    fund: r.classList.contains('fund'), w: parseFloat(r.querySelector('.chs-t i').style.width) })); }"""


def test_the_bars_are_the_shares_the_overview_publishes(page):
    ov = page.evaluate("(D.history && D.history.overview) || {}")
    bases = [b for b in ("prev", "m1") if (ov.get(b) or {}).get("shares")]
    if not bases:
        pytest.skip("MISSING DATA: the published history carries no category shares")
    labels = page.evaluate("CAT_LABELS")
    for base in bases:
        shares = ov[base]["shares"]
        rows = page.evaluate(_SHARES_JS, ov[base])
        want = sorted((k for k in shares if k in labels), key=lambda k: -shares[k])
        assert [r["n"] for r in rows] == [labels[k] for k in want], base
        for r, k in zip(rows, want):
            assert r["v"] == page.evaluate("v => pctNearZero(v * 100, 1).mag", shares[k]) + "%"
            assert r["fund"] == (k in run_overview.FUNDAMENTAL)
        widths = [r["w"] for r in rows]
        assert widths == sorted(widths, reverse=True) and widths[0] == pytest.approx(100)
    page.evaluate("renderChanged()")


def test_no_shares_draws_nothing(page):
    assert page.evaluate("(() => { renderChangedShares(null); return document.getElementById('changed-shares').innerHTML; })()") == ""
    page.evaluate("renderChanged()")


# --------------------------------------------------------------------------- the track record, period by period
_LEAD_JS = """(T) => { const d = document.createElement('div'); d.innerHTML = trackLeadBars(T);
    return { up: d.querySelectorAll('.tr-lb-plot i.up').length, dn: d.querySelectorAll('.tr-lb-plot i.dn').length,
             cls: [...d.querySelectorAll('.tr-lb-plot i')].map(i => i.className), text: d.textContent }; }"""


def test_the_green_bars_are_the_published_count_of_periods_ahead(page):
    T = page.evaluate("D.track || null")
    if not T or not T.get("available"):
        pytest.skip("MISSING DATA: the published context carries no track record")
    got = page.evaluate(_LEAD_JS, T)
    if T["periods_compared"] < 2:
        assert got["up"] + got["dn"] == 0
        return
    assert got["up"] == T["beat_rsp"]
    assert got["up"] + got["dn"] == T["periods_compared"]
    assert f"Ahead in {T['beat_rsp']} of {T['periods_compared']}" in got["text"]


def test_a_tie_is_not_drawn_as_a_lead(page):
    """The record counts a period ahead only when the top 25 beat RSP (``top > RSP``); a tie is not."""
    T = {"periods": [{"date": "2026-01-02", "entry": "2026-01-02", "exit": "2026-02-02", "top": 0.02, "RSP": 0.01},
                     {"date": "2026-02-02", "entry": "2026-02-02", "exit": "2026-03-02", "top": 0.01, "RSP": 0.01},
                     {"date": "2026-03-02", "entry": "2026-03-02", "exit": "2026-04-02", "top": -0.01, "RSP": 0.0},
                     {"date": "2026-04-02", "entry": "2026-04-02", "exit": "2026-05-02", "top": None, "RSP": 0.0}],
         "beat_rsp": 1, "periods_compared": 3}
    got = page.evaluate(_LEAD_JS, T)
    assert got["cls"] == ["up", "dn", "dn"]


# --------------------------------------------------------------------------- the sector table's standouts
def test_each_marked_cell_is_its_columns_unique_extreme(page):
    page.evaluate("updateSectorDist()")
    cols = page.evaluate("""() => { const t = document.querySelector('#sector-matrix table'); if (!t) return null;
        const heads = [...t.querySelectorAll('thead th')].map(th => th.textContent);
        return heads.map((h, i) => ({ h, cells: [...t.querySelectorAll('tbody tr')].map(r => {
            const td = r.children[i]; return { v: parseFloat(td.textContent), hi: td.classList.contains('sm-hi'), lo: td.classList.contains('sm-lo') }; }) })); }""")
    if not cols:
        pytest.skip("MISSING DATA: no sector distributions published")
    for c in cols:
        vals = [x["v"] for x in c["cells"] if x["v"] == x["v"]]
        his = [x for x in c["cells"] if x["hi"]]
        los = [x for x in c["cells"] if x["lo"]]
        if c["h"] in ("Sector", "Composite", "Stocks") or not vals:
            assert not his and not los, c["h"]
            continue
        assert len(his) <= 1 and len(los) <= 1, c["h"]
        if his:
            assert his[0]["v"] == max(vals) and vals.count(max(vals)) == 1, c["h"]
        if los:
            assert los[0]["v"] == min(vals) and vals.count(min(vals)) == 1, c["h"]
    # no floor on how many are marked: a day where every column ties at its extremes marks nothing,
    # and that is correct (this check runs in the 02:00 publish gate - row "data->red")


# --------------------------------------------------------------------------- the analyst-target bar's end labels
def test_the_target_range_labels_stay_inside_their_bar(page):
    t = examples.stock("MU")
    page.evaluate("t => openStockDetail(t)", t)
    page.wait_for_timeout(600)
    got = page.evaluate("""() => { const lab = document.querySelector('#stock-modal .pt-range-labels'); if (!lab) return null;
        const L = lab.getBoundingClientRect();
        return [...lab.children].map(s => { const r = s.getBoundingClientRect(); return [r.left - L.left, L.right - r.right]; }); }""")
    page.evaluate("closeModal()")
    if not got:
        pytest.skip("MISSING DATA: the example stock carries no analyst target range")
    for left_gap, right_gap in got:
        assert left_gap >= -0.5 and right_gap >= -0.5, got
