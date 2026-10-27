"""What the options panel tells a reader about the published research behind its numbers.

Why this exists
---------------
``research/2026-10-12-option-implied-signals.md`` reviewed all four numbers on the panel and
left three findings open for a later session (``plan/context-layer.md`` item 4). All three are
about a sentence rather than a score, and all three are the same failure: **a figure from a
paper printed beside a number the paper did not measure.**

1. The panel quoted Xing, Zhang & Zhao's (2010) ~11% a year next to a skew this screener did
   not compute. Theirs is the IV of an out-of-the-money put minus that of the at-the-money
   **call**; ours took the put nearest 0.90 moneyness and subtracted the **mean** of the ATM
   call and put - which is their skew plus half of Cremers & Weinbaum's (2010) put-call IV
   spread, so it was neither published quantity. The computation now follows the paper
   (``context_signals._xzz_skew``), and both legs are printed so the subtraction is checkable.
2. Only the one-week figure was quoted. The paper's **Table 4 Panel B** measures the same
   spread at a four-week holding period (6.52% a year, t 2.70) and from 8 to 28 weeks (6-7%,
   t 3.73-4.31). A reader on a monthly review cadence can act on the four-week number and not
   on the one-week one, so the panel gives both.
3. No decay figure was given, where one exists: McLean & Pontiff (2016) measure returns 58%
   lower post-publication across 97 predictors.

And the fourth number - put/call **open interest** - had no sentence at all. It is a bare
*level*, where the published evidence concerns flow: Pan & Poteshman (2006) find publicly
observable option volume has "very little, if any" predictability, and XZZ's own control for
the put-call volume ratio is significant one week ahead (t 3.62) and at no horizon from four
weeks out (Table 4 Panel A). So it is off the page, and still in the context log.

All figures here were read from the papers, not from a survey:
``data/papers/xzz-2010.pdf``, ``mp-2016-a.pdf`` (the forthcoming-JF manuscript, which carries
the published 97-predictor / 26% / 58% figures - the 2012-13 working papers say 82 and 35%),
``pp-2006-a.pdf``. The download scripts are in ``research/measurements/``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import claims  # noqa: E402
import context_signals as cs  # noqa: E402
import generate_dashboard as gd  # noqa: E402


def _panel() -> str:
    """The context panel as the generator writes it - never the committed page (rule 10)."""
    return gd._js_context()


# --------------------------------------------------------------------------- the construction
def test_the_panel_states_the_papers_construction():
    js = _panel()
    assert "Xing, Zhang &amp; Zhao" in js
    assert "out-of-the-money put nearest 95% of the price" in js
    assert "at-the-money call" in js, "the benchmark leg is the ATM call, and must be named"
    # The superseded wording must not come back: it described a measure nobody published.
    assert "90% puts vs at the money" not in js


def test_the_constants_are_the_papers():
    """Literal pins only. Renamed 2026-10-26 from ``..._and_the_code_uses_them``: the body never
    tested use, and a name that claims more than the body checks is how row 44 of
    ``plan/claims-mutation-audit.md`` hid. That the code *uses* each band is tested on its
    edges in ``tests/test_context_layer.py`` (``test_skew_*``)."""
    assert (cs.XZZ_PUT_LO, cs.XZZ_PUT_HI, cs.XZZ_PUT_TARGET) == (0.80, 0.95, 0.95)
    assert (cs.XZZ_CALL_LO, cs.XZZ_CALL_HI, cs.XZZ_CALL_TARGET) == (0.95, 1.05, 1.0)
    assert (cs.XZZ_DAYS_LO, cs.XZZ_DAYS_HI) == (10, 60)


def test_the_skew_row_prints_both_legs():
    """Row 0.8: a number on the page is checkable from the page."""
    js = _panel()
    row = js[js.index("Put skew"):]
    for key in ("c.oskp", "c.oskc", "c.oskps", "c.oskcs"):
        assert key in row, f"the skew row does not print {key}, so the subtraction cannot be checked"


# --------------------------------------------------------------------------- the effect sizes
def test_the_panel_quotes_the_four_week_figure_not_only_the_one_week_one():
    js = _panel()
    assert "10.9%" in js, "the paper's one-week quintile alpha spread"
    assert "6.5%" in js, "the paper's four-week holding-period spread (Table 4 Panel B: 6.52%)"
    assert "four" in js and "one week" in js


def test_the_panel_names_the_papers_option_window():
    js = _panel()
    assert "10 to 60 days" in js
    assert "c.oskw" in js, "the panel must say when this expiry falls outside that window"


def test_the_panel_states_the_post_publication_shrinkage():
    js = _panel()
    assert "58%" in js and "97 published predictors" in js
    assert "McLean &amp; Pontiff" in js
    assert "upper bound" in js, "the gaps are a ceiling, not an expectation"


def test_the_panel_does_not_promise_the_effect_holds_now():
    """The superseded sentence said gaps were 'smaller in later data' with no figure behind it."""
    assert "with smaller gaps in later data" not in _panel()


# --------------------------------------------------------------------------- put/call open interest
def test_put_call_open_interest_is_not_published():
    assert "_ctx_opt_pc_oi" not in gd.CTX_FIELDS.values(), (
        "a bare put/call open-interest level is back in the payload; the evidence (Pan & "
        "Poteshman 2006; XZZ Table 4 Panel A) is about flow and is spent within a week"
    )
    js = _panel()
    assert "Put / call open interest" not in js
    assert "c.opc" not in js


def test_the_context_log_still_records_it_so_the_evidence_keeps_accruing():
    """Off the page is not out of the record - context_eval must still evaluate the signal."""
    src = (ROOT / "context_eval.py").read_text(encoding="utf-8")
    assert "_ctx_opt_pc_oi" in src and "options_put_call_oi" in src


# --------------------------------------------------------------------------- publish as a unit
def _ctx(extra: dict) -> dict:
    """One stock's published context, from a raw row."""
    row = dict({"_ctx_last_close": 100.0, "_ctx_opt_status": "ok",
                "_ctx_opt_expiry": "2026-11-13", "_ctx_opt_days": 30}, **extra)
    return gd._stock_context(row, "TEST", "2026-10-14") or {}


def test_a_skew_without_its_legs_is_not_published():
    """A run fetched before 2026-10-14 carries the superseded measure in ``_ctx_opt_skew``.

    The panel's sentence names the paper's construction, so a number whose legs are missing
    cannot be shown under it - it was produced by different arithmetic. Withholding the row is
    the "say what cannot be shown" rule; it returns by itself at the next full fetch.
    """
    out = _ctx({"_ctx_opt_skew": 0.08, "_ctx_opt_skew_strike": 90.0})
    assert "osk" not in out, "the old-construction skew is still being published"
    assert "oskps" not in out and "oskw" not in out
    # The rest of the options panel is unaffected - only the skew row goes.
    assert out.get("os") == "ok"


def test_a_skew_with_both_legs_is_published_whole():
    out = _ctx({"_ctx_opt_skew": 0.08, "_ctx_opt_skew_strike": 90.0,
                "_ctx_opt_skew_put_iv": 0.38, "_ctx_opt_skew_call_strike": 100.0,
                "_ctx_opt_skew_call_iv": 0.30, "_ctx_opt_skew_in_window": True})
    assert out["osk"] == pytest.approx(0.08)
    assert out["oskp"] == pytest.approx(0.38) and out["oskc"] == pytest.approx(0.30)
    assert out["oskps"] == 90.0 and out["oskcs"] == 100.0 and out["oskw"] is True
    # The legs reproduce the number shown, which is what the row claims on its face.
    assert out["oskp"] - out["oskc"] == pytest.approx(out["osk"])


# --------------------------------------------------------------------------- the register
def test_the_claim_is_registered_and_resolves():
    claim = next((c for c in claims.CLAIMS if c.id == "context.put_skew"), None)
    assert claim is not None, "the skew sentence is not in the claims register"
    assert claim.resolve() is cs._xzz_skew
    for name in claim.checked_by:
        mod, _, test = name.partition("::")
        assert (ROOT / mod).exists(), f"{claim.id} names a missing test module: {mod}"
        assert re.search(rf"def {re.escape(test)}\b", (ROOT / mod).read_text(encoding="utf-8")), \
            f"{claim.id} names a test that does not exist: {name}"
