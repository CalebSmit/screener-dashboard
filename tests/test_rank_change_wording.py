"""A rank change of one says "1 rank" (2026-11-06).

The stock sheet's rank history heads each category table with the rank change over that span.
HST's sheet that night read "▼1 ranks". The generator's own page, with the published payloads;
the change is swapped in and restored, so the test asserts the wording, not which stock moved.
"""
from __future__ import annotations

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
except Exception:  # noqa: BLE001
    sync_api = None


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("rankwording")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
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
        pg.wait_for_function("typeof renderStockHistory === 'function' && typeof H !== 'undefined'", timeout=60000)
        sheet_warm.settle(pg)
        yield pg
        b.close()
        assert not errors, errors


_JS = """(dr) => {
    const t = Object.keys(H.delta || {}).find(k => H.delta[k] && H.delta[k].prev && H.delta[k].prev.cat
        && !H.delta[k].prev.new && H.series && H.series[k]);
    if (!t) return null;
    const e = H.delta[t].prev, keep = e.dr;
    try { e.dr = dr; renderStockHistory(t);
          return document.querySelector('#modal-history .modal-hist-head span:last-child').textContent.trim(); }
    finally { e.dr = keep; } }"""


@pytest.mark.parametrize("dr, want", [(1, "▲1 rank"), (-1, "▼1 rank"), (2, "▲2 ranks"),
                                      (-37, "▼37 ranks"), (0, "unchanged")])
def test_the_rank_change_is_counted_in_words_that_agree(page, dr, want):
    got = page.evaluate(_JS, dr)
    if got is None:
        pytest.skip("MISSING DATA: no stock carries a since-last-run category table")
    assert got == want
