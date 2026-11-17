"""Does a web font that arrives late defeat `_settled` in tests/test_lab_reading_and_access.py?

The 2026-11-17 06:00 baseline was red on
`test_the_height_the_card_bound_reads_waits_for_the_page_and_sees_a_taller_card`:
`sum(settled) == 4549` against `sum(before) + 500 == 4459`. The 90 px between them is the
figure that module's own docstring records for Inter replacing the fallback face on the
experiment cards (89 px, 2026-11-09), so the hypothesis is that `before` was read in the
fallback face and the font landed afterwards.

Prediction, written before running: adding DELAY seconds of latency to every network request
(the stylesheet, then each font file, so the font lands after about twice that) makes the helper named on the command line return the fallback-face height (lower than
the height read 6 s later) whenever DELAY exceeds the helper's own patience, and the same
height when it does not.

    python research/measurements/2026-11-17-late-font-defeats-settled.py [delay_seconds ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from playwright.sync_api import sync_playwright  # noqa: E402

import test_lab_reading_and_access as t  # noqa: E402

CARDS = "[...document.querySelectorAll('#o-exp ~ .cards .card')].map(c => c.offsetHeight)"


def trial(browser, delay: float) -> tuple[int, int, float]:
    ctx, page, _ = t._page(browser)
    try:
        if delay:
            # latency on network requests only: file:// is not throttled, so the page and its
            # data arrive at once and only Google Fonts' stylesheet and font files are late.
            # (A sleeping route handler will not do - it blocks the thread the helper runs on.)
            cdp = ctx.new_cdp_session(page)
            cdp.send("Network.enable")
            cdp.send("Network.emulateNetworkConditions", {
                "offline": False, "latency": delay * 1000,
                "downloadThroughput": -1, "uploadThroughput": -1})
        t._goto(page, "overview")
        t0 = time.time()
        got = sum(t._settled(page, CARDS))
        took = time.time() - t0
        page.wait_for_timeout(6000)
        later = sum(page.evaluate(CARDS))
        return got, later, took
    finally:
        ctx.close()


if __name__ == "__main__":
    delays = [float(a) for a in sys.argv[1:]] or [0.0, 1.0, 3.0]
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for d in delays:
            got, later, took = trial(b, d)
            print(f"latency {d:>4.1f}s: _settled={got}  6s later={later}  "
                  f"{'SAME' if got == later else 'EARLY by ' + str(later - got) + ' px'}  "
                  f"(helper took {took:.1f}s)")
        b.close()
