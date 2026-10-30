"""Every stock's drilldown, opened on the generator's own page, reads without a broken value.

Added 2026-10-29 after two defects of one kind in two days, each in a formatter no test reached:
BXP's case-against line printed "0.0%" for a close 0.03% below its 200-day average (the data
loop's claims gate refused that night's run over it), and the next evening a sweep of all 498
panels on the live page found AMD's five-year earnings-yield range reading "-0.0% – 3.2%" and
INTC's median "-0.0%" (``cPlain``). The sweep found nothing else, so it is kept as a test: it opens
every published stock through ``openStockDetail``, the function a reader's click calls, and fails
on any line carrying a value the JavaScript could not format.

It reads the committed payloads, so a data run that publishes a new glitch turns this red for the
next code session, which is the session that should fix it. It is deliberately not in the data
loop's publish gate (``scripts/check_published_claims.py``): a cosmetic glitch should cost a fix,
not a day of evidence.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

try:
    from playwright import sync_api
except ImportError:  # pragma: no cover - skips loudly without it
    sync_api = None

# A value the page could not format. Each alternative is something no correct sentence prints:
# JavaScript's names for a missing or impossible value, an object coerced to text, a signed zero
# (a small negative rounded away - the defect class above), and an empty pair of brackets.
BROKEN = r"\b(?:undefined|NaN|Infinity|null)\b|\[object Object\]|[−-]0\.0+%|\(\s*\)"

SWEEP = r"""(pattern) => {
  const pat = new RegExp(pattern);
  const root = () => { let e = document.getElementById('modal-ticker');
    while (e && e.parentElement && e.innerText.length < 5000) e = e.parentElement; return e; };
  const hits = [], failed = [];
  for (const r of D.table_data) {
    try {
      openStockDetail(r.Ticker);
      const bad = root().innerText.split('\n').filter(l => pat.test(l));
      if (bad.length) hits.push([r.Ticker, bad.slice(0, 3).map(l => l.slice(0, 160))]);
    } catch (e) { failed.push([r.Ticker, String(e).slice(0, 160)]); }
  }
  closeModal();
  return {n: D.table_data.length, hits, failed};
}"""


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    if sync_api is None:
        pytest.skip("MISSING DEP: playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("panels")
    (d / "index.html").write_text(
        gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v"),
        encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        pg = b.new_page(viewport={"width": 375, "height": 812})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        # the context panels are part of what a reader opens, so wait for them to land
        pg.wait_for_function("typeof openStockDetail === 'function' && typeof D !== 'undefined' && "
                             "Object.values(D.stock_detail).some(s => s.ctx)", timeout=60000)
        out = pg.evaluate(SWEEP, BROKEN)
        b.close()
    out["page_errors"] = errors
    return out


def test_every_published_stock_was_opened(sweep):
    assert sweep["n"] >= 490, sweep["n"]
    assert not sweep["failed"], sweep["failed"][:5]
    assert not sweep["page_errors"], sweep["page_errors"][:5]


def test_no_panel_prints_a_value_the_page_could_not_format(sweep):
    assert not sweep["hits"], f"{len(sweep['hits'])} panels: {sweep['hits'][:8]}"


@pytest.mark.parametrize("line, broken", [
    ("Five-year range -0.0% – 3.2%", True),
    ("median −0.0% (dashed line)", True),
    ("Earnings yield undefined", True),
    ("a ratio of NaN", True),
    ("Five-year range -0.02% – 3.2%", False),
    ("median 0.0% (dashed line)", False),
    ("Its last close is −10.0% against its 200-day average", False),
    ("Sustainable Growth (0.0%), in the 6th sector percentile.", False),
    ("Nullable fields are named (as filed)", False),
])
def test_the_pattern_tells_a_broken_value_from_a_real_one(line, broken):
    assert bool(re.search(BROKEN, line)) is broken
