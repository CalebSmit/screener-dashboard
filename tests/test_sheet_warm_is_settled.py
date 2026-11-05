"""A browser test that renders into the stock sheet must settle the sheet pre-warm first.

WHY THIS EXISTS - 2026-11-05. The 13:00 session's gate-1 baseline reported exactly one failure,
`tests/test_context_layer_claims.py::test_the_page_says_when_each_kind_of_quote_was_taken`, with no
assertion and no traceback (the baseline saved no output then; it does now). The same command on
the same commit reported 3,976 passed seven minutes later, and it never reproduced in 8 further
runs across 4 scenarios, two of them whole-suite parallel runs
(`research/measurements/2026-11-05-context-claims-flake.py`).

The cause, measured rather than inferred
(`research/measurements/2026-11-05-prewarm-clobbers-the-panel.py`): `generate_dashboard`'s
`sheetPrewarm` lays the top-ranked stock's sheet out unseen once the page is idle, by calling
`renderSheet(top, ...)` - which **rewrites `#modal-context`**. Tests that render synthetic context
into the sheet do it in two round-trips, a render then a read, and a fixture's own
`wait_for_function` returns with `SHEET_WARM.state === 'wait'`. Forced, the panel comes back as the
real top stock's: the synthetic sentence is gone and "close of" is present, which is exactly the
pair of assertions that failed that morning.

Nothing in production is wrong - a reader's `openStockDetail` stops the pre-warm and rewrites the
sheet - so the fix is in the harness (`tests/sheet_warm.py`), and this is the tripwire that keeps
it. **It is the reason the module to watch is the next one written, not the three fixed today:**
the flake first appeared on the day a session added `tests/test_dashboard_printed_figures.py`,
which renders per-stock context in a loop and so opens the window hundreds of times a run.

Two rules this module keeps for itself, from `plan/claims-mutation-audit.md` row 59 - a check may
not import the thing it is checking:

* it holds its **own** copy of the state expression rather than importing `sheet_warm.SETTLE_JS`,
  so widening the helper cannot widen the assertion;
* the behavioural test drives the page itself rather than reusing any module's fixture.
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
except Exception:  # noqa: BLE001
    sync_api = None

TESTS = ROOT / "tests"

# What makes a module exposed: it writes into the sheet from JS, or reads the panel the pre-warm
# rewrites, without going through an open (which stops the pre-warm itself).
RENDERS = ("renderContext(", "renderSheet(", "modal-context")
OPENS = ("_open_stock", "openStockDetail", "examples.stock")
SETTLES = ("sheet_warm.settle", "SHEET_WARM")


def _browser_modules() -> list[Path]:
    return [p for p in sorted(TESTS.glob("test_*.py"))
            if "playwright" in p.read_text(encoding="utf-8", errors="replace")]


def test_every_module_that_renders_the_sheet_settles_the_prewarm():
    """The sweep. A module may be exempt only by opening a stock, which settles it in the page."""
    offenders = []
    for p in _browser_modules():
        t = p.read_text(encoding="utf-8", errors="replace")
        if not any(m in t for m in RENDERS):
            continue
        if any(m in t for m in OPENS) or any(m in t for m in SETTLES):
            continue
        offenders.append(p.name)
    assert not offenders, (
        f"these browser modules render into the stock sheet without settling the pre-warm, so "
        f"renderSheet() can overwrite what they rendered from an idle callback: {offenders}. "
        f"Call tests.sheet_warm.settle(page) in the fixture, after its wait_for_function - see "
        f"tests/sheet_warm.py."
    )


def test_the_three_modules_fixed_today_still_settle_it():
    """Named deliberately. The sweep above passes for a module that opens a stock, and a future
    edit could add an open to one of these three while dropping the settle - which would read as
    covered and would not be."""
    for name in ("test_context_layer_claims.py", "test_context_cards_printed.py",
                 "test_dashboard_printed_figures.py"):
        t = (TESTS / name).read_text(encoding="utf-8")
        assert "sheet_warm.settle(pg)" in t, name
        assert "from tests import sheet_warm" in t, name
        # after the fixture's wait, never before it: settling a pre-warm that has not started is
        # right, but the page must be ready enough for stop() to exist
        assert t.index("wait_for_function") < t.index("sheet_warm.settle(pg)"), name


def test_the_helper_cannot_pass_on_a_prewarm_that_is_still_pending():
    """`settle` accepts 'done' and 'absent' and nothing else - in particular not 'wait' or 'run',
    which are exactly the states the clobber happens in."""
    from tests import sheet_warm

    class FakePage:
        def __init__(self, state):
            self.state = state

        def evaluate(self, _js):
            return self.state

    assert sheet_warm.settle(FakePage("done")) == "done"
    assert sheet_warm.settle(FakePage("absent")) == "absent"
    for bad in ("wait", "run", None, ""):
        with pytest.raises(AssertionError):
            sheet_warm.settle(FakePage(bad))


# --------------------------------------------------------------------------- the behaviour itself
@pytest.fixture(scope="module")
def fresh_page(tmp_path_factory):
    """A factory for pages with the pre-warm deliberately *pending*, one per call.

    A factory rather than one page, because the two things worth checking need different starting
    states: one lets the pre-warm run to see that it really does overwrite a rendered panel, and
    the other settles it **while it is still pending**, which is the only state the helper has to
    work in. Reusing the first page for the second would settle an already-finished pre-warm and
    pass whatever the helper did - measured: with `settle` reduced to a no-op that whole test still
    passed (`research/measurements/2026-11-05-sheet-warm-tripwire-mutations.py`).
    """
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("warm")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")

        def make():
            pg = b.new_page(viewport={"width": 1280, "height": 900})
            pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
            pg.wait_for_function(
                "typeof renderContext === 'function' && typeof D !== 'undefined' && "
                "Object.values(D.stock_detail).some(s => s.ctx)", timeout=60000)
            return pg

        yield make
        b.close()


# This module's own copy, not the helper's.
STATE = "typeof SHEET_WARM === 'undefined' ? 'absent' : SHEET_WARM.state"
SYNTHETIC = {"os": "ok", "om": 0.05, "ox": "2026-11-20", "od": 31, "px": 100.0, "oi": 0.30,
             "oqd": "2026-10-20", "oqt": "14:21", "oqi": True}
SENTENCE = "Quotes were taken at 14:21 ET on Oct 20, while the market was open"


def _panel(page) -> str:
    return re.sub(r"\s+", " ", page.evaluate("document.getElementById('modal-context').textContent"))


def test_the_prewarm_really_does_overwrite_a_rendered_panel(fresh_page):
    """The window is real. Two facts, both relationships rather than today's data:

    1. a fixture that waits only for the payload leaves the pre-warm pending - which is why every
       test that renders into the sheet is exposed, including the three fixed today;
    2. letting it run replaces a rendered panel with the pre-warm's own stock.

    No date, stock or figure is named: step 2 asserts the panel stopped holding what was rendered
    into it and that the sheet now names the ticker the pre-warm chose.
    """
    page = fresh_page()
    assert page.evaluate(STATE) in ("wait", "run"), (
        "a fixture that waits only for the payload must leave the pre-warm pending, or this whole "
        "class of flake is gone and this module should be reconsidered"
    )
    page.evaluate("s => renderContext('ZZZ', s)", {"sector": "Energy", "raw": {}, "ctx": SYNTHETIC})
    assert SENTENCE in _panel(page), "the synthetic render must land before anything else"

    page.wait_for_function("typeof SHEET_WARM === 'undefined' || SHEET_WARM.state === 'done'",
                           timeout=60000)
    warmed = page.evaluate("typeof SHEET_WARM === 'undefined' ? null : (SHEET_WARM.ticker || null)")
    if warmed is None:
        pytest.skip("this build has no sheet pre-warm, so there is no window to close")
    assert SENTENCE not in _panel(page), (
        "the pre-warm did not overwrite the panel - if that is now impossible by construction, "
        "tests/sheet_warm.py and this module can go; until then the helper is what makes the "
        "three modules using it deterministic"
    )
    assert page.evaluate("document.getElementById('modal-ticker').textContent") == warmed
    page.close()


def test_settling_a_pending_prewarm_makes_the_render_survive(fresh_page):
    """The helper, exercised in the only state it has to work in: a **pending** pre-warm, on a
    page that has just loaded - which is where every fixture calls it.

    Separate from the test above, and on its own page, deliberately. Settling an already-finished
    pre-warm passes whatever the helper does, so a single-page version of this test was blind to
    `settle` being reduced to a no-op.
    """
    page = fresh_page()
    if page.evaluate(STATE) == "absent":
        pytest.skip("this build has no sheet pre-warm")
    assert page.evaluate(STATE) in ("wait", "run")

    from tests import sheet_warm
    assert sheet_warm.settle(page) == "done"
    page.evaluate("s => renderContext('ZZZ', s)", {"sector": "Energy", "raw": {}, "ctx": SYNTHETIC})
    assert SENTENCE in _panel(page)
    # past the pre-warm's own deadline (600 ms + fonts + an idle slice, retried up to 20 x 250 ms)
    page.wait_for_timeout(2500)
    assert SENTENCE in _panel(page), "settled, nothing may rewrite the panel later"
    assert page.evaluate(STATE) == "done"
    page.close()
