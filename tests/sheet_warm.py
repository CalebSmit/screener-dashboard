"""Settle the sheet pre-warm before a test renders into the stock sheet.

WHY THIS EXISTS - 2026-11-05. `generate_dashboard`'s `sheetPrewarm` lays the top-ranked stock's
sheet out unseen once the page is idle, so the first open is not a cold layout. It gets there by
calling `renderSheet(top, ...)`, which **rewrites `#modal-context`** - and it fires about 600 ms
after `load`, then after `document.fonts.ready`, then on an idle slice. For a reader that is
correct and invisible: `openStockDetail` stops the pre-warm and rewrites the sheet anyway.

For a browser test that renders synthetic context straight into the sheet it is a race, because
the usual helper is two round-trips:

    page.evaluate("s => renderContext('ZZZ', s)", s)                        # write
    page.evaluate("document.getElementById('modal-context').textContent")   # read

If the pre-warm lands between them, the text read is the real top stock's context. Measured in
`research/measurements/2026-11-05-prewarm-clobbers-the-panel.py`: forced, the panel comes back as
VLO's, the synthetic sentence is gone and "close of" is present - which is exactly the pair of
assertions that failed in the 13:00 session's gate-1 baseline that morning
(`tests/test_context_layer_claims.py::test_the_page_says_when_each_kind_of_quote_was_taken`, one
failure, never reproduced in 8 further runs). A fixture's own wait returns with
`SHEET_WARM.state === 'wait'`, so the window is open for every test the fixture serves.

A module that opens a stock does not need this - `openStockDetail` stops the pre-warm itself.

`tests/test_sheet_warm_is_settled.py` is the tripwire: a module that renders the sheet without
settling fails the suite. That module deliberately keeps its own copy of the state expression, and
so does `tests/test_sheet_round_three.py`, which *tests* the pre-warm - a check may not import the
thing it is checking (`plan/claims-mutation-audit.md`, row 59).
"""
from __future__ import annotations

# Stop it if it is running, and move it out of 'wait' so it can never start. `stop()` is the page's
# own function: it clears `sheet-warm`/`warm-skip` and sets the state to 'done'.
SETTLE_JS = """() => {
    if (typeof SHEET_WARM === 'undefined') return 'absent';
    if (SHEET_WARM.stop) SHEET_WARM.stop(); else SHEET_WARM.state = 'done';
    return SHEET_WARM.state;
}"""


def settle(page) -> str:
    """Settle the sheet pre-warm on `page`. Returns its state ('done', or 'absent' on a page that
    has none - a stand-in DOM, or a build where the pre-warm was removed).

    Call it once per page, after the fixture's `wait_for_function` and before the first render.
    Idempotent, and never fatal on a page without a pre-warm: a helper that can only fail closed
    would stop a test from running at all over a performance feature it does not care about.
    """
    state = page.evaluate(SETTLE_JS)
    assert state in ("done", "absent"), (
        f"the sheet pre-warm did not settle (state {state!r}); until it does, anything this test "
        f"renders into the sheet can be overwritten by renderSheet() on an idle callback"
    )
    return state
