"""What to check before trusting a high rank (``ranking_audit.py``, 2026-10-15).

Pinned: a stock whose net income runs well above its after-tax operating income, and one whose
equity buybacks have thinned to under a tenth of its assets with sustainable growth at its cap,
each get a plain sentence on the metric affected; a note already there is never replaced; the
top-25 line names the reasons; the sentences carry no advice; and no scoring module reads any of it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ranking_audit as RA  # noqa: E402


def _detail(ni=1.027e9, ebit=0.893e9, eq=6.4e9, ta=13e9, sgr=0.06, **extra):
    d = {"inp": {"netIncome": ni, "ebit": ebit, "totalEquity": eq, "totalAssets": ta},
         "raw": {"sustainable_growth": sgr}, "pct": {}}
    d.update(extra)
    return d


def test_earnings_lifted_by_income_outside_operations_are_said():
    notes = RA.stock_notes(_detail())                       # Host Hotels' figures, 2026-10-15
    s = notes["earnings_yield"]
    assert "1.46 times operating income" in s and "$1.03bn" in s and "may not recur" in s
    assert "sustainable_growth" not in notes


def test_a_buyback_thinned_equity_base_is_said_when_the_measure_is_at_its_cap():
    d = _detail(ni=2.036e9, ebit=2.738e9, eq=1.209e9, ta=28.8e9, sgr=1.0)   # Expedia
    notes = RA.stock_notes(d)
    assert "4.2% of total assets" in notes["sustainable_growth"] and "100% cap" in notes["sustainable_growth"]
    assert "earnings_yield" not in notes
    assert "sustainable_growth" not in RA.stock_notes(_detail(eq=1.209e9, ta=28.8e9, sgr=0.4))  # not at the cap


def test_ordinary_figures_get_no_note_and_losses_are_left_alone():
    assert RA.stock_notes(_detail(ni=0.6e9, ebit=0.9e9)) == {}
    assert RA.stock_notes(_detail(ni=-1e9, ebit=0.9e9)) == {}
    assert RA.stock_notes(_detail(ni=1e9, ebit=-0.5e9)) == {}
    assert RA.stock_notes({"inp": {}, "raw": {}}) == {}


def test_the_notes_have_their_own_field_and_heading_and_leave_the_filed_note_alone():
    d = _detail(cavs={"earnings_yield": "filed-figure note"})
    sd = {"HST": d}
    assert RA.attach(sd) == 1
    assert sd["HST"]["cavs"] == {"earnings_yield": "filed-figure note"}
    assert "may not recur" in sd["HST"]["fign"]["earnings_yield"]
    import generate_dashboard as g
    html = g.generate_html()
    assert "In this company&rsquo;s reported figures." in html and "(s.fign || {})[m]" in html


def test_the_top_25_line_names_its_reasons():
    payload = {
        "table_data": [{"Ticker": "A", "Rank": 1}, {"Ticker": "B", "Rank": 2}, {"Ticker": "C", "Rank": 30}],
        "stock_detail": {"A": _detail(), "B": _detail(ni=0.5e9, xchk={"ni": 0.16, "ta": 0.0},
                                                         newer_filing={"end": "2026-06-30"}),
                         "C": _detail()},
    }
    flags = RA.top_flags(payload)
    assert [f["ticker"] for f in flags] == ["A", "B"]       # C is outside the top 25
    assert flags[0]["why"] == ["net income 1.46x after-tax operating income"]
    assert flags[1]["why"] == ["net income 16% off the filing", "a period behind its filings"]


def test_the_sentences_carry_no_advice():
    from stock_summary import advice_terms_in
    text = " ".join(RA.stock_notes(_detail()).values()) + " " + " ".join(
        RA.stock_notes(_detail(eq=1e8, ta=28.8e9, sgr=1.0)).values())
    assert advice_terms_in(text) == []


def test_no_scoring_module_reads_it():
    pat = re.compile(r"^\s*(import ranking_audit|from ranking_audit)\b", re.M)
    for name in ("factor_engine.py", "run_screener.py", "portfolio_constructor.py", "calc_trace.py",
                 "improvement_engine.py", "backtest.py"):
        assert not pat.search((ROOT / name).read_text(encoding="utf-8", errors="replace")), name


def test_the_page_build_and_the_health_check_both_run_it():
    gd = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert "ranking_audit.attach(stock_detail)" in gd
    assert gd.index("_attach_capex_lease_note(stock_detail)") < gd.index("ranking_audit.attach(stock_detail)")
    hc = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
    assert "check_top_of_ranking(payload, res)" in hc
    sys.path.insert(0, str(ROOT / "scripts"))
    import write_brief as wb
    assert wb.TOP_LINE == RA.TOP_LINE
    assert wb.top_of_ranking(["[02:20] ok   top 25, worth a look: HST (#3: x)"]) == "HST (#3: x)"
    assert wb.top_of_ranking([]) == ""
