"""The Diagnostics tile says what its bars are (claim ``dashboard.stability_tile``, 2026-11-06).

The tile drew each category's published top-20 stability (``weight_sensitivity[].avg_jaccard``,
1.0 = unchanged) on a 0-1 scale under a dashed line at the view's own 0.85 "Robust" bar. On that
scale 0.70 and 0.86 are a few pixels apart, so the eight bars read as one height and nothing on the
tile said what they measured. Each bar at or above the line is now marked, and a caption says what
the bars are and how many reach it. These tests hold the caption and the marks to the payload and
the rule to hand-worked boundaries.
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
from tests import sheet_warm  # noqa: E402

try:
    from playwright import sync_api
except ImportError:  # pragma: no cover
    sync_api = None

LINE = 0.85


def _payload(name: str) -> dict:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"MISSING DATA: no published {name}")
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))


def _caption(ok: int, n: int) -> str:
    return (f"Top-20 overlap when each weight moves 5 points; {ok} of {n} "
            f"{'reaches' if ok == 1 else 'reach'} the dashed 0.85 line, where the list barely changes")


def test_the_line_is_the_views_own_robust_bar():
    """The tile's line and the Diagnostics view's 'Robust' threshold are one number, and the
    engine's perturbation is the 5 points the caption names."""
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert f"const STAB_LINE = {LINE};" in src
    assert "avgJ >= 0.85 ? 'Robust'" in src
    import inspect
    import factor_engine
    sig = inspect.signature(factor_engine.run_weight_sensitivity)
    assert sig.parameters["perturbation_pct"].default == 5.0
    assert sig.parameters["top_n"].default == 20


def test_the_diagnostics_view_says_percentage_points_not_percent():
    """The view's description said each weight 'moved ±5%' - the wording the methodology page dropped
    on 2026-10-19 (claim overview.weight_sensitivity): for Size's 5% it reads as 4.75-5.25, where the
    run tries 0 and 10. Read from the generator's own page, not a committed file."""
    import run_screener as rs
    html = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    pts = rs._pts(rs.SENSITIVITY_PERTURBATION_PTS)
    assert f"weight moved {pts} percentage points up and down" in html
    assert f"moves {pts} points;" in html
    assert f"the top {rs.SENSITIVITY_TOP_N} compared" in html
    for bad in ("&plusmn;5%", "±5%"):
        assert bad not in html, bad


@pytest.fixture(scope="module")
def data() -> dict:
    return _payload("dashboard_data.js")


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("stab")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
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
        pg.wait_for_function("typeof stabilityTileText === 'function' && typeof D !== 'undefined'", timeout=60000)
        sheet_warm.settle(pg)
        yield pg
        pg.close()
        b.close()
        assert not errors, errors


def _tile(page, sens):
    return page.evaluate("""sens => { const keep = D.weight_sensitivity; D.weight_sensitivity = sens;
        try { renderTileViz(); const v = document.querySelector('#sec-defensibility .tile-viz');
              return v && { cap: v.querySelector('.tv-stab-cap').textContent,
                            ok: [...v.querySelectorAll('.tv-stab i')].map(i => i.classList.contains('ok')),
                            label: v.getAttribute('aria-label') }; }
        finally { D.weight_sensitivity = keep; renderTileViz(); } }""", sens)


def test_the_tile_prints_the_published_figures(page, data):
    sens = [x for x in data.get("weight_sensitivity") or [] if isinstance(x.get("avg_jaccard"), (int, float))]
    if not sens:
        pytest.skip("MISSING DATA: the published payload carries no weight sensitivity")
    got = _tile(page, data["weight_sensitivity"])
    ok = [x["avg_jaccard"] >= LINE for x in sens]
    assert got["ok"] == ok
    assert got["cap"] == _caption(sum(ok), len(sens))
    assert got["label"].endswith(_caption(sum(ok), len(sens)))


def test_the_line_counts_at_exactly_085_and_not_below(page):
    sens = [{"category": "Valuation", "avg_jaccard": 0.85}, {"category": "Quality", "avg_jaccard": 0.8499},
            {"category": "Growth", "avg_jaccard": 1.0}, {"category": "Size", "avg_jaccard": None}]
    got = _tile(page, sens)
    assert got["ok"] == [True, False, True]
    assert got["cap"] == _caption(2, 3)
    one = _tile(page, sens[:2])
    assert one["cap"] == _caption(1, 2)


def test_the_view_prints_overlap_on_the_scale_of_its_own_bars(page, data):
    """The badge and every row read on the 0-1 scale the 0.85 and 0.70 bars are stated in, and a
    weight's move is in percentage points - the table read "+5%: 82%" until 2026-11-06."""
    sens = [s for s in data.get("weight_sensitivity") or [] if isinstance(s.get("avg_jaccard"), (int, float))]
    if not sens:
        pytest.skip("MISSING DATA: no weight_sensitivity in the published payload")
    got = page.evaluate("""() => { renderDefensibility && renderDefensibility();
        return { badge: document.querySelector('#defensibility-summary .def-badge').textContent.trim(),
                 rows: [...document.querySelectorAll('#sensitivity-table .sens-bar-labels')].map(e => e.textContent.trim()) }; }""")
    mean = sum(s["avg_jaccard"] for s in sens) / len(sens)
    verdict = "robust" if mean >= 0.85 else "moderate" if mean >= 0.70 else "sensitive"
    assert got["badge"] == f"Stability {verdict} ({_jsfixed(mean, 2)})", got["badge"]
    import run_screener as rs
    pts = rs._pts(rs.SENSITIVITY_PERTURBATION_PTS)
    want = [f"+{pts}pp: {_jsfixed(s['plus_jaccard'], 2)}−{pts}pp: {_jsfixed(s['minus_jaccard'], 2)}"
            for s in data["weight_sensitivity"] if s.get("avg_jaccard") is not None]
    assert got["rows"] == want, (got["rows"], want)


def _jsfixed(x: float, n: int) -> str:
    from decimal import Decimal, ROUND_HALF_UP
    # the double's exact value, as toFixed rounds it (0.825 is 0.82499... and prints 0.82)
    return str(Decimal(float(x)).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))
