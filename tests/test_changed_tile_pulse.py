"""The What Changed tile carries the panel's two figures for the last run (claim
``history.tile_pulse``, 2026-11-06).

The owner's question on the day the S&P 500 fell 4.4% was whether the ranking held. The panel
answered it (``renderChangedPulse``); the front-page tile did not, so a reader had to know to open
it. The tile now ends with the same two figures for the last run - the index's move between the two
runs' closes (``history.compare.prev.market``) and the rank correlation of the two rankings
(``history.overview.prev.rho``), with the typical run-to-run figure - and nothing it does not have.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
from tests import sheet_warm  # noqa: E402

try:
    from playwright import sync_api
except ImportError:  # pragma: no cover
    sync_api = None


def _payload(name: str) -> dict:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"MISSING DATA: no published {name}")
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


def _want(ov: dict | None, cmp: dict | None) -> str:
    """The tile's line as text, from the history's own figures."""
    rho = (ov or {}).get("rho")
    if not isinstance(rho, (int, float)) or cmp is None:
        return ""
    m = (cmp or {}).get("market") or {}
    mkt = ""
    if isinstance(m.get("ret"), (int, float)):
        fig = f"{abs(m['ret']) * 100:.1f}"
        sign = "" if fig == "0.0" else ("−" if m["ret"] < 0 else "+")
        mkt = f"S&P 500 {sign}{fig}%, "
    typ = ov.get("rho_typical")
    return (f"Since the last run: {mkt}rank correlation {rho:.3f}"
            + (f" (typically {typ:.3f})" if isinstance(typ, (int, float)) else ""))


@pytest.fixture(scope="module")
def data() -> dict:
    return _payload("dashboard_data.js")


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("pulse_tile")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        pg = b.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof changedTilePulse === 'function' && typeof H !== 'undefined' && "
                             "document.querySelector('#sec-changed .tile-viz') !== null", timeout=60000)
        sheet_warm.settle(pg)
        yield pg
        pg.close()
        b.close()
        assert not errors, errors


def _tile(page, ov, cmp):
    """The tile's pulse line and colour class with H.overview.prev / H.compare.prev replaced."""
    return page.evaluate("""([ov, cmp]) => {
        const keepO = H.overview, keepC = H.compare;
        H.overview = Object.assign({}, keepO, { prev: ov }); H.compare = Object.assign({}, keepC, { prev: cmp });
        try { renderTileViz(); const p = document.querySelector('#sec-changed .tile-viz .tv-pulse');
              const b = p && p.querySelector('b.dn, b.up');
              return { text: p ? p.textContent : '', tone: b ? b.className : '',
                       label: document.querySelector('#sec-changed .tile-viz').getAttribute('aria-label') }; }
        finally { H.overview = keepO; H.compare = keepC; renderTileViz(); } }""", [ov, cmp])


def test_the_tile_prints_the_published_figures(page, data):
    h = data.get("history") or {}
    ov, cmp = (h.get("overview") or {}).get("prev"), (h.get("compare") or {}).get("prev")
    if not (h.get("available") and (h.get("movers") or {})):
        pytest.skip("MISSING DATA: the published history has no movers, so the tile is not drawn")
    got = page.evaluate("""() => { const p = document.querySelector('#sec-changed .tile-viz .tv-pulse');
                                   return p ? p.textContent : ''; }""")
    assert got == _want(ov, cmp)


CASES = [
    ({"rho": 0.987, "rho_typical": 0.996},
     {"date": "2026-11-05", "market": {"ret": -0.04356, "from": "2026-11-04", "to": "2026-11-05"}}, "dn"),
    ({"rho": 0.99, "rho_typical": 0.996}, {"date": "2026-11-05", "market": {"ret": 0.0123}}, "up"),
    ({"rho": 0.999, "rho_typical": None}, {"date": "2026-11-05", "market": {"ret": 0.0004}}, ""),
    ({"rho": 0.95, "rho_typical": 0.996}, {"date": "2026-11-05"}, ""),
    ({"rho": None}, {"date": "2026-11-05", "market": {"ret": -0.02}}, ""),
]


@pytest.mark.parametrize("ov,cmp,tone", CASES)
def test_the_tile_on_hand_worked_figures(page, ov, cmp, tone):
    got = _tile(page, ov, cmp)
    assert got["text"] == _want(ov, cmp), (got, ov, cmp)
    assert got["tone"] == tone
    if got["text"]:
        assert got["label"].endswith(". " + got["text"].replace("−", "-"))


def test_a_fall_reads_as_a_fall():
    want = _want(CASES[0][0], CASES[0][1])
    assert want == "Since the last run: S&P 500 −4.4%, rank correlation 0.987 (typically 0.996)"
    assert re.search(r"\+0\.0|−0\.0", _want(CASES[2][0], CASES[2][1])) is None
