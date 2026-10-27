"""A quarter the source never published is rolled forward, not invented (``_ltm_rollforward``).

Measured 2026-10-15 on Berkshire Hathaway, whose 2025-09-30 quarter the source does not carry:
``_stmt_val_ltm`` could not sum four adjacent quarters, so it annualised three by 4/3 and the
published run scored net income 14.5% below, operating cash flow 6.1% below and revenue 10.2%
above the company's own filed twelve months. The twelve months was available all along from the
same two statements, by the identity the filer uses in its own 10-Q:

    TTM(E) = fiscal year + year-to-date(E) - year-to-date(E one year earlier)

which reproduces BRK-B's filed net income, operating cash flow and capital expenditure to the
dollar (``research/measurements/2026-10-15-ttm-rollforward.py``).

These tests hold the construction, its guards, the precedence against the exact four-quarter sum,
and that the route taken is recorded on the stock. The two facts that must not drift are pinned
in ``test_the_identity_matches_the_measured_berkshire_case``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402

# Berkshire's published quarters on 2026-10-15: 2025-09-30 is absent, not blank.
QS = [pd.Timestamp(d) for d in ("2026-06-30", "2026-03-31", "2025-12-31",
                                "2025-06-30", "2025-03-31")]
AS = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")]
LAB = "Net Income"


def _q(values, cols=None):
    cols = cols if cols is not None else QS
    return pd.DataFrame({c: [v] for c, v in zip(cols, values)}, index=[LAB])


def _a(values, cols=None):
    return pd.DataFrame({c: [v] for c, v in zip(cols if cols is not None else AS, values)},
                        index=[LAB])


# ---------------------------------------------------------------- the construction

def test_the_identity_is_the_fiscal_year_plus_the_stub_less_the_year_before():
    # FY 100; the two quarters since it 30 + 40; the same two a year earlier 20 + 25.
    q = _q([40, 30, 999, 25, 20])          # 2025-12-31 is the FY column, never read from here
    v, how = fe._ltm_rollforward(q, _a([100, 90]), LAB)
    assert v == 100 + 70 - 45
    assert "fiscal year to 2025-12-31" in how and "2 quarter(s) to 2026-06-30" in how


def test_one_stub_quarter_and_three_stub_quarters_both_work():
    one = _q([10, 7], cols=[pd.Timestamp("2026-03-31"), pd.Timestamp("2025-03-31")])
    v, _h = fe._ltm_rollforward(one, _a([100, 90]), LAB)
    assert v == 100 + 10 - 7
    cols3 = [pd.Timestamp(d) for d in ("2026-09-30", "2026-06-30", "2026-03-31",
                                       "2025-09-30", "2025-06-30", "2025-03-31")]
    three = _q([12, 11, 10, 9, 8, 7], cols=cols3)
    v3, _h3 = fe._ltm_rollforward(three, _a([100, 90]), LAB)
    assert v3 == 100 + 33 - 24


def test_the_identity_matches_the_measured_berkshire_case():
    """The arithmetic that produced $85.768bn, against the figures the source served.

    Yahoo's annual net income for BRK-B was $66.968bn to 2025-12-31; its quarters to
    2026-06-30 and 2026-03-31 and the two a year earlier are the stubs. The result equals
    Berkshire's own filed ``NetIncomeLoss`` for the twelve months to 2026-06-30 to the dollar,
    where 3 of 4 quarters times 4/3 gave $73.296bn - 14.5% low.
    """
    q = _q([24.274e9, 16.335e9, np.nan, 12.698e9, 4.603e9])
    v, _how = fe._ltm_rollforward(q, _a([66.968e9, 89.0e9]), LAB)
    assert round(v / 1e9, 3) == round((66.968 + 24.274 + 16.335 - 12.698 - 4.603), 3)


# ---------------------------------------------------------------- the guards

def test_a_gap_between_the_fiscal_year_end_and_the_stub_is_refused():
    # the stub starts two quarters after the fiscal year end: the identity would skip a quarter
    cols = [pd.Timestamp(d) for d in ("2026-06-30", "2025-06-30")]
    v, how = fe._ltm_rollforward(_q([40, 25], cols=cols), _a([100, 90]), LAB)
    assert np.isnan(v) and how is None


def test_a_missing_year_earlier_quarter_is_refused():
    q = _q([40, 30, 999, 25], cols=QS[:4])          # no 2025-03-31 to pair with 2026-03-31
    v, how = fe._ltm_rollforward(q, _a([100, 90]), LAB)
    assert np.isnan(v) and how is None


def test_a_stub_that_does_not_reach_the_newest_quarter_is_refused():
    # the newest published quarter is before the fiscal year end: there is no stub to add
    q = _q([30, 25], cols=[pd.Timestamp("2025-09-30"), pd.Timestamp("2024-09-30")])
    v, how = fe._ltm_rollforward(q, _a([100, 90]), LAB)
    assert np.isnan(v) and how is None


def test_four_or_more_stub_quarters_are_refused():
    cols = [pd.Timestamp(d) for d in ("2026-12-31", "2026-09-30", "2026-06-30", "2026-03-31",
                                      "2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31")]
    v, how = fe._ltm_rollforward(_q([13, 12, 11, 10, 9, 8, 7, 6], cols=cols), _a([100, 90]), LAB)
    assert np.isnan(v) and how is None


def test_missing_statements_and_missing_labels_are_refused_quietly():
    for q, a in [(None, _a([100, 90])), (_q([40, 30, 1, 25, 20]), None),
                 (pd.DataFrame(), _a([100, 90])), (_q([40, 30, 1, 25, 20]), pd.DataFrame())]:
        v, how = fe._ltm_rollforward(q, a, LAB)
        assert np.isnan(v) and how is None
    v, how = fe._ltm_rollforward(_q([40, 30, 1, 25, 20]), _a([100, 90]), "Nonexistent Line")
    assert np.isnan(v) and how is None


# ---------------------------------------------------------------- precedence and the record

def test_four_adjacent_quarters_still_win_and_the_annual_is_not_consulted():
    full = [pd.Timestamp(d) for d in ("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30")]
    q = _q([40, 30, 20, 10], cols=full)
    rolled: list[str] = []
    partial: list[str] = []
    v = fe._stmt_val_ltm(q, LAB, annual=_a([999, 888]), rolled_labels=rolled,
                         partial_labels=partial)
    assert v == 100 and rolled == [] and partial == []


def test_the_roll_forward_is_preferred_over_annualising_three_quarters():
    q = _q([40, 30, 20, 25, 20])                    # 4 columns, not adjacent (no 2025-09-30)
    rolled: list[str] = []
    partial: list[str] = []
    v = fe._stmt_val_ltm(q, LAB, annual=_a([100, 90]), rolled_labels=rolled,
                         partial_labels=partial)
    assert v == 100 + 70 - 45
    assert rolled == [LAB] and partial == []
    # without the annual statement the old behaviour is unchanged
    v43 = fe._stmt_val_ltm(q, LAB, partial_labels=partial)
    assert v43 == (40 + 30 + 20) * (4 / 3) and partial == [LAB]


def test_the_prior_period_ltm_never_rolls_forward():
    """offset=4 is the prior-year window; the identity is only defined for the latest."""
    q = _q([40, 30, 20, 25, 20])
    rolled: list[str] = []
    fe._stmt_val_ltm(q, LAB, offset=4, annual=_a([100, 90]), rolled_labels=rolled)
    assert rolled == []


class _FakeTicker:
    """Enough of yfinance for the statement blocks of the fetch, with Berkshire's hole."""

    def __init__(self):
        self.info = {"symbol": "BRK-B", "marketCap": 1.1e12, "currentPrice": 500.0}
        rows = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
        self.quarterly_financials = pd.DataFrame(
            {QS[0]: [110e9, 30e9, 20e9, 24e9], QS[1]: [100e9, 28e9, 18e9, 16e9],
             QS[2]: [105e9, 29e9, 19e9, 20e9], QS[3]: [95e9, 27e9, 17e9, 13e9],
             QS[4]: [90e9, 26e9, 16e9, 5e9]}, index=rows)
        self.financials = pd.DataFrame({AS[0]: [400e9, 110e9, 70e9, 67e9],
                                        AS[1]: [380e9, 105e9, 68e9, 89e9]}, index=rows)
        self.quarterly_cashflow = pd.DataFrame(
            {QS[0]: [13e9], QS[1]: [12e9], QS[2]: [11e9], QS[3]: [10e9], QS[4]: [9e9]},
            index=["Operating Cash Flow"])
        self.cashflow = pd.DataFrame({AS[0]: [46e9], AS[1]: [44e9]},
                                     index=["Operating Cash Flow"])

    def __getattr__(self, name):
        return None


def test_the_fetch_rolls_forward_and_records_which_lines_it_rolled(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda *a, **k: _FakeTicker())
    rec = fe._fetch_single_ticker_inner("BRK-B")
    assert rec["_ltm_rolled"] is True
    assert rec["_ltm_annualized"] is False          # nothing was annualised
    assert "Net Income" in rec["_ltm_rolled_labels"]
    assert rec["netIncome"] == 67e9 + (24e9 + 16e9) - (13e9 + 5e9)
    assert rec["operatingCashFlow"] == 46e9 + (13e9 + 12e9) - (10e9 + 9e9)
    assert rec["_data_source"] == "quarterly"


def test_every_current_period_ltm_call_site_is_given_its_annual_statement():
    """A call site left without ``annual=`` silently keeps the 4/3 fabrication."""
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    body = src[src.index("def _fetch_single_ticker_inner"):]
    missed = [line.strip() for line in body.splitlines()
              if "_stmt_val_ltm(q_" in line and "offset=" not in line
              and "annual=" not in line]
    assert missed == [], missed
    assert body.count("annual=fins") >= 12 and body.count("annual=cf") >= 9


def _generator_function(name: str) -> str:
    """One JS function's source out of the generator, by brace depth."""
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    i = src.index(f"function {name}(")
    depth, j = 0, src.index("{", i)
    k = j
    while True:
        if src.startswith("{{", k):
            depth += 1
            k += 2
        elif src.startswith("}}", k):
            depth -= 1
            k += 2
            if depth == 0:
                return src[i:k]
        else:
            k += 1


def test_the_badge_states_the_identity_the_engine_uses():
    """The panel may not describe a different construction from the one in the code."""
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert '"ltm_rolled": bool(row.get("_ltm_rolled"))' in src
    assert '"ltm_rolled_labels": _str_list(row.get("_ltm_rolled_labels"))' in src
    fn = _generator_function("renderProvenance")
    i = fn.index("if (fl.ltm_rolled)")
    sentence = fn[i:i + 1100]
    for phrase in ("Twelve months rolled forward",
                   "had not published one of this company",
                   "last fiscal year plus the quarters since it",
                   "less the same quarters a year earlier",
                   "its own 10-Q",
                   "fl.ltm_rolled_labels"):
        assert phrase in sentence, phrase
    # `listed` is used by this sentence and is a const: it must be declared before it
    assert fn.index("const listed = function") < i


def test_the_provenance_sentences_are_in_the_function_the_page_calls():
    """Found 2026-10-15: ``renderFlagsWarnings`` is dead - never called, and its container
    ``#modal-provenance``'s sibling ``#modal-flags`` does not exist in the page the generator
    writes. The quarterly-rejection badge added that morning sat in it and could never appear.
    Both provenance sentences now live in ``renderProvenance``, which the drilldown does call.
    """
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    prov = _generator_function("renderProvenance")
    assert "if (fl.q_rejected)" in prov and "if (fl.ltm_rolled)" in prov
    assert "renderProvenance(" in src.replace("function renderProvenance(", "", 1)
    assert "id=\"modal-provenance\"" in src or "'modal-provenance'" in src
    # The dead function was deleted on 2026-10-27 once each of its eleven badges had been
    # accounted for (`tests/test_dead_badges.py`). What this test still has to enforce is that
    # nothing re-creates a container-less sink for a badge: the whole defect was that a badge
    # written into one cannot fail visibly.
    assert "function renderFlagsWarnings" not in src, (
        "renderFlagsWarnings is back - it had no container, so anything in it is invisible; "
        "put the badge in a function the drilldown calls (see tests/test_dead_badges.py)")
    assert "getElementById('modal-flags')" not in src and 'id="modal-flags"' not in src


def test_the_panel_does_not_contradict_itself_about_how_the_twelve_months_was_built():
    """The general rule may not be stated flatly above a sentence saying it did not apply."""
    prov = _generator_function("renderProvenance")
    i = prov.index("const ltmLead")
    lead = prov[i:prov.index("Prices are the last close", i)]
    assert "fl.ltm_rolled || fl.q_rejected" in lead
    assert "built for this company as described below" in lead
    assert "the sum of the last four reported quarters" in lead   # kept for every other stock
    # and the exception sentences come after the lead, so "below" is true
    assert i < prov.index("if (fl.q_rejected)") and i < prov.index("if (fl.ltm_rolled)")


def test_the_methodology_page_states_the_roll_forward():
    """Generated, so assert the generator - never the committed file (CLAUDE.md rule 10)."""
    src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    i = src.index("A quarter the source never published")
    para = src[i:src.index("\n", i)]
    for phrase in ("the quarters since it", "the same quarters a year earlier",
                   "its own 10-Q", "14.5%", "sum × 4/3", "_ltm_rolled"):
        assert phrase in para, phrase
    # the old text claimed 4/3 was the only route
    assert "the screener annualizes (sum × 4/3) but flags" not in src


def test_the_roll_forward_never_reads_a_filing():
    """Scoring does not touch the SEC cache (CLAUDE.md 0.14) - the cache measures this, not feeds it."""
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    fn = src[src.index("def _ltm_rollforward"):src.index("def _stmt_val_ltm")]
    for banned in ("sec_crosscheck", "sec_fundamentals", "companyfacts", "xchk", "pit_record"):
        assert banned not in fn
