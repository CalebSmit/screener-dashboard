"""Can the sheet pre-warm overwrite a panel a browser test just rendered? Measured, not argued.

WHY. On 2026-11-05 the 13:00 session's gate-1 baseline reported exactly one failure,
`tests/test_context_layer_claims.py::test_the_page_says_when_each_kind_of_quote_was_taken`, and the
same command on the same commit reported 3,976 passed seven minutes later. It never reproduced in
8 runs across 4 scenarios (`2026-11-05-context-claims-flake.py`).

THE HYPOTHESIS. `generate_dashboard`'s `sheetPrewarm` lays the top-ranked stock's sheet out unseen
once the page is idle - deliberate, and right for a reader, because `openStockDetail` stops it and
rewrites the sheet. It gets there by calling `renderSheet(top, ...)`, which **rewrites
`#modal-context`**. The test's helper is two round-trips:

    page.evaluate("s => renderContext('ZZZ', s)", s)            # writes synthetic ZZZ context
    page.evaluate("document.getElementById('modal-context').textContent")   # reads it back

and the module's `page` fixture yields as soon as the payload is in, while the pre-warm is still
pending (600 ms after load, then `document.fonts.ready`, then an idle slice). If the pre-warm fires
between those two calls, the text read is the *real* top stock's context, whose quotes come from
after the close - so "while the market was open" is absent and "close of" is present, which is
exactly the pair of assertions that test makes first.

WHAT THIS SCRIPT DOES. Three measurements, no assertions about timing:

  1. Is `SHEET_WARM` observable from `page.evaluate`, and what does it do on its own? (If it is not
     observable the fix cannot wait on it.)
  2. **The clobber, forced.** Render synthetic ZZZ context, then wait for the pre-warm to reach
     'done', then read the panel. If the hypothesis holds, the text is the top stock's, the
     synthetic sentence is gone, and the two assertions that failed this morning fail here.
  3. **The fix, checked.** Settle the pre-warm *before* rendering, then repeat. The synthetic text
     must survive, and `SHEET_WARM.state` must be 'done' so nothing can fire later.

    python research/measurements/2026-11-05-prewarm-clobbers-the-panel.py
"""
from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

# the helper from tests/test_context_layer_claims.py, verbatim in substance
BASE = {"os": "ok", "om": 0.05, "ox": "2026-11-20", "od": 31, "px": 100.0, "oi": 0.30}
MID = {**BASE, "oqd": "2026-10-20", "oqt": "14:21", "oqi": True}
WANTED = "Quotes were taken at 14:21 ET on Oct 20, while the market was open"


def options_text(page, ctx):
    page.evaluate("s => renderContext('ZZZ', s)", {"sector": "Energy", "raw": {}, "ctx": ctx})
    return re.sub(r"\s+", " ", page.evaluate("document.getElementById('modal-context').textContent"))


def main() -> int:
    from playwright import sync_api

    d = Path(tempfile.mkdtemp(prefix="prewarm-"))
    (d / "index.html").write_text(
        gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v"),
        encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            print(f"MISSING DATA: no published {f} - nothing to measure")
            return 3
        shutil.copy(ROOT / f, d / f)

    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        try:
            for label in ("forced", "fixed"):
                pg = b.new_page(viewport={"width": 1280, "height": 900})
                pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
                # the fixture's own wait, verbatim: it returns while the pre-warm is pending
                pg.wait_for_function(
                    "typeof renderContext === 'function' && typeof D !== 'undefined' && "
                    "Object.values(D.stock_detail).some(s => s.ctx)", timeout=60000)

                if label == "forced":
                    observable = pg.evaluate("typeof SHEET_WARM")
                    print(f"[1] typeof SHEET_WARM from page.evaluate: {observable!r}")
                    print(f"    state right after the fixture's wait: "
                          f"{pg.evaluate('typeof SHEET_WARM === \"undefined\" ? null : SHEET_WARM.state')!r}")
                    if observable == "undefined":
                        print("    not observable - a fix cannot wait on it; stop here")
                        return 1

                    print("\n[2] the clobber, forced: render ZZZ, let the pre-warm finish, then read")
                    pg.evaluate("s => renderContext('ZZZ', s)",
                                {"sector": "Energy", "raw": {}, "ctx": MID})
                    pg.wait_for_function("SHEET_WARM.state === 'done'", timeout=30000)
                    text = re.sub(r"\s+", " ",
                                  pg.evaluate("document.getElementById('modal-context').textContent"))
                    warmed = pg.evaluate("SHEET_WARM.ticker || null")
                    print(f"    pre-warm rendered: {warmed}")
                    print(f"    synthetic sentence still present: {WANTED in text}")
                    print(f"    \"close of\" present (the test asserts it is not): {'close of' in text}")
                    print(f"    panel now reads: {text[:170]}...")
                    print("    => the two assertions that failed at 13:03 both fail here"
                          if WANTED not in text and "close of" in text else
                          "    => NOT reproduced this way")
                else:
                    print("\n[3] the fix: settle the pre-warm before rendering, then render and read")
                    pg.evaluate("() => { if (typeof SHEET_WARM !== 'undefined' && SHEET_WARM.stop) "
                                "SHEET_WARM.stop(); }")
                    state = pg.evaluate("SHEET_WARM.state")
                    text = options_text(pg, MID)
                    print(f"    SHEET_WARM.state after stop(): {state!r}")
                    print(f"    synthetic sentence present: {WANTED in text}")
                    print(f"    \"close of\" absent: {'close of' not in text}")
                    pg.wait_for_timeout(2500)   # nothing may fire later
                    text2 = options_text(pg, MID)
                    print(f"    still true 2.5 s later: "
                          f"{WANTED in text2 and 'close of' not in text2}")
                pg.close()
        finally:
            b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
