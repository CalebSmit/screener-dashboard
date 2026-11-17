"""Figures read from the annual statement are dated by it, and the stock's page says so (2026-11-17).

The pre-registered re-run of the filing cross-check's range table that morning found operating
cash flow over 10% from the filing for 10 stocks where it had been 1-3 until 2026-11-02. For
nine of them the source's quarterly cash-flow statement carried 2026-09-30, 2026-03-31,
2025-12-31, ... - no June quarter - so four quarters could not be summed, the last fiscal year
could not be rolled forward (the stub is not consecutive), no three adjacent quarters could be
scaled, and the fetch read the annual statement. That part is right. What was wrong is that the
statement's **date** was still taken from the quarterly statement's newest column: Amazon,
Boeing and Bank of America carried fiscal-2025 income and cash flow dated 2026-09-30. So

- the panel said "Income statement as of Sep 30, 2026" over figures ending 2025-12-31;
- the staleness check read 48 days for figures 321 days old;
- `sec_crosscheck.newer_filings`, built on 2026-10-15 for exactly these companies, reported
  "0 stocks scored a period behind a filing" every morning, because it reads that date;
- and the filing cross-check compared a fiscal year with a different twelve months.

`research/measurements/2026-11-17-annual-figures-under-a-quarterly-date.py` counts it: 15 of 418
stocks on operating cash flow, 17 of 427 on net income.

No score moves: the figures were already the annual statement's. What is pinned here is the date,
the record of which statement was replaced, and the sentence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402
import generate_dashboard as g  # noqa: E402
import sec_crosscheck as X  # noqa: E402

T = pd.Timestamp
# the shape measured on 2026-11-17: the June quarter is not there
GAPPED = [T(d) for d in ("2026-09-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30")]
WHOLE = [T(d) for d in ("2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30")]
YEARS = [T("2025-12-31"), T("2024-12-31")]
IS_ROWS = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
CF_ROWS = ["Operating Cash Flow", "Capital Expenditure"]


class _Ticker:
    """Enough of yfinance for the statement blocks of the fetch."""

    def __init__(self, is_cols=GAPPED, cf_cols=GAPPED, quarterly=True):
        self.info = {"symbol": "TEST", "marketCap": 1e11, "currentPrice": 100.0}
        if quarterly:
            self.quarterly_financials = pd.DataFrame(
                {c: [10e9 + i * 1e8, 4e9, 2e9, 1e9 + i * 1e7] for i, c in enumerate(is_cols)}, index=IS_ROWS)
            self.quarterly_cashflow = pd.DataFrame(
                {c: [3e9 + i * 1e8, -1e9] for i, c in enumerate(cf_cols)}, index=CF_ROWS)
            self.quarterly_balance_sheet = pd.DataFrame({is_cols[0]: [50e9], is_cols[1]: [48e9]},
                                                        index=["Total Assets"])
        self.financials = pd.DataFrame({YEARS[0]: [38e9, 15e9, 7e9, 3.5e9],
                                        YEARS[1]: [35e9, 14e9, 6e9, 3.0e9]}, index=IS_ROWS)
        self.cashflow = pd.DataFrame({YEARS[0]: [11e9, -4e9], YEARS[1]: [10e9, -3e9]}, index=CF_ROWS)

    def __getattr__(self, name):
        return None


def _fetch(monkeypatch, **kw):
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda *a, **k: _Ticker(**kw))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fe._fetch_single_ticker_inner("TEST")


def test_the_premise_a_gapped_statement_supports_none_of_the_three_routes():
    """Asserted rather than assumed: with the June quarter absent nothing makes twelve months."""
    t = _Ticker()
    partial, rolled = [], []
    v = fe._stmt_val_ltm(t.quarterly_cashflow, "Operating Cash Flow", partial_labels=partial,
                         annual=t.cashflow, rolled_labels=rolled)
    assert np.isnan(v) and partial == [] and rolled == []


def test_figures_from_the_annual_statement_carry_its_date_not_the_newest_quarters(monkeypatch):
    rec = _fetch(monkeypatch)
    # the figures are the fiscal year's - unchanged behaviour, and the reason the date matters
    assert rec["netIncome"] == 3.5e9 and rec["operatingCashFlow"] == 11e9 and rec["capex"] == -4e9
    assert rec["_data_source"] == "annual"
    # ... so the date is the fiscal year's too. Both read "2026-09-30" before 2026-11-17.
    assert rec["_stmt_date_financials"] == "2025-12-31"
    assert rec["_stmt_date_cashflow"] == "2025-12-31"
    # the balance sheet really is the newest quarter's, and still says so
    assert rec["_stmt_date_balance_sheet"] == "2026-09-30"
    assert rec["_annual_instead"] == "income statement and cash-flow statement"


def test_each_statement_is_dated_on_its_own(monkeypatch):
    """Only the cash-flow statement gapped: income stays the twelve months to the newest quarter."""
    rec = _fetch(monkeypatch, is_cols=WHOLE, cf_cols=GAPPED)
    assert rec["_data_source"] == "quarterly"
    assert rec["netIncome"] == sum(1e9 + i * 1e7 for i in range(4))
    assert rec["operatingCashFlow"] == 11e9
    assert rec["_stmt_date_financials"] == "2026-09-30"
    assert rec["_stmt_date_cashflow"] == "2025-12-31"
    assert rec["_annual_instead"] == "cash-flow statement"


def test_a_stock_whose_quarters_sum_is_untouched(monkeypatch):
    rec = _fetch(monkeypatch, is_cols=WHOLE, cf_cols=WHOLE)
    assert rec["_annual_instead"] is None and rec["_data_source"] == "quarterly"
    assert rec["_stmt_date_financials"] == rec["_stmt_date_cashflow"] == "2026-09-30"
    assert rec["operatingCashFlow"] == sum(3e9 + i * 1e8 for i in range(4))


def test_a_company_with_no_quarterly_statements_is_not_said_to_have_unsummable_ones(monkeypatch):
    """`_annual_instead` records quarters that were on file and could not be used - a company the
    source serves only annual statements for is the old, already-described case."""
    rec = _fetch(monkeypatch, quarterly=False)
    assert rec["_data_source"] == "annual" and rec["_annual_instead"] is None
    assert rec["_stmt_date_financials"] == "2025-12-31"


def test_the_newer_filing_note_now_sees_these_stocks():
    """The consequence that mattered most: the 2026-10-15 note reads the income statement's date.
    Dated by the quarter, a stock on fiscal-year figures was invisible to it; dated by the
    fiscal year, the company's filed 10-Q for the later period is named."""
    facts = pd.DataFrame([
        {"ticker": "A", "concept": "Revenues", "start": T("2025-01-01"), "end": T("2025-12-31"),
         "filed": T("2026-02-05"), "form": "10-K", "fp": "FY", "val": 4.0},
        {"ticker": "A", "concept": "Revenues", "start": T("2026-07-01"), "end": T("2026-09-30"),
         "filed": T("2026-10-30"), "form": "10-Q", "fp": "Q3", "val": 1.0},
    ])
    det = lambda d: {"kpis": {"run_timestamp": "2026-11-17T02:00:00"},  # noqa: E731
                     "stock_detail": {"A": {"inp": {}, "asof": {"is": d}}}}
    assert X.newer_filings(det("2026-09-30"), facts) == {}
    assert X.newer_filings(det("2025-12-31"), facts) == {
        "A": {"end": "2026-09-30", "filed": "2026-10-30", "form": "10-Q"}}


def test_the_staleness_check_reads_the_age_of_the_figures(monkeypatch):
    """`_stmt_age_days` is taken from the income statement's date in `compute_metrics`: from the
    quarterly date it understated these stocks' age by the nine months between the two."""
    rec = _fetch(monkeypatch)
    gap = (T("2026-09-30") - T(rec["_stmt_date_financials"])).days
    assert gap == 273


# ------------------------------------------------------------------------------- the page
def _panel() -> str:
    """`renderProvenance` as the generated page carries it (the slice `test_newer_filing` uses)."""
    html = g.generate_html()
    return html[html.index("function renderProvenance(s)"):html.index("// INIT")]


def test_the_generator_publishes_the_fetchs_own_record_and_never_scores_it():
    assert "_annual_instead" in g.PROVENANCE_KEYS
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert '"annual_instead": row.get("_in__annual_instead")' in src
    # display-only: nothing that scores, traces or constructs reads it
    for name in ("calc_trace.py", "portfolio_constructor.py", "metric_lineage.py", "pit_record.py"):
        assert "annual_instead" not in (ROOT / name).read_text(encoding="utf-8", errors="replace"), name
    eng = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    body = eng[eng.index("def compute_metrics"):]
    assert "_annual_instead" not in body, "the scorer must not read the fetch's provenance record"


def test_the_panel_says_the_figures_are_the_last_fiscal_years_and_why():
    panel = _panel()
    i = panel.index('id="provenance-annual-instead"')
    sentence = panel[i:i + 900]
    for phrase in ("Last fiscal year used.",
                   "not carry four consecutive quarters with a figure each",
                   "neither rolling the last fiscal year forward nor scaling three quarters up",
                   "the latest annual statement",
                   "the fiscal year ending on the date given above",
                   "escapeHtml(fl.annual_instead)"):
        assert phrase in sentence, phrase
    # and the general rule is not stated flatly above it
    lead = panel[panel.index("const ltmLead"):panel.index("Prices are the last close")]
    assert "fl.annual_instead" in lead
    # a rejected quarterly statement has its own sentence; the two never both print
    assert "if (fl.annual_instead && !fl.q_rejected)" in panel


def test_the_period_behind_note_does_not_blame_the_source_for_a_period_it_did_publish():
    """For these stocks the source *has* the newest quarter - it lacks an earlier one - so "the
    data source had not published that period" would be false on their pages. The note is
    written in two places (the provenance panel and the case summary) and both must branch."""
    html = g.generate_html()
    hits = [html[m:m + 700] for m in
            (i for i in range(len(html)) if html.startswith("for the period to ' +", i))]
    assert len(hits) == 2, len(hits)
    for h in hits:
        assert "fl.annual_instead && !fl.q_rejected" in h
        assert "last fiscal year" in h and "had not published that period" in h


def test_the_methodology_page_states_the_fourth_route():
    """Generated, so assert the generator - never the committed file (CLAUDE.md rule 10)."""
    import run_screener
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    text = run_screener.build_screener_overview(cfg)
    i = text.index("A quarter the source never published")
    para = text[i:text.index("\n", i)]
    for phrase in ("Where neither route is available", "the latest annual statement is used",
                   "dates it by the fiscal year the figures cover", "never by the newest quarter on file",
                   "_annual_instead"):
        assert phrase in para, phrase
