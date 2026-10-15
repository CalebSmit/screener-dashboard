"""Data Provenance says which of a stock's inputs agree with its own SEC filing.

WHY THIS EXISTS - 2026-10-14. The panel used to end "it cannot check that they are true". For
most stocks it now can: the run's inputs are rebuilt from the company's XBRL filings for the
same quarters (sec_crosscheck.py; research/2026-10-14-yahoo-vs-sec.md) and the drilldown says
which agree within 2% and which differ, with the usual reasons for a difference.

Pinned: the payload carries the check as display-only provenance (never in raw/pct, never in a
score), a build without the SEC cache simply omits it, and the panel's sentence uses the 2% line
the claim register states (claim ``drilldown.filing_check``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as g  # noqa: E402


def _facts():
    T = pd.Timestamp
    return pd.DataFrame([
        {"ticker": "A", "concept": "Assets", "start": pd.NaT, "end": T("2025-12-31"), "filed": T("2026-02-10"),
         "form": "10-K", "fp": "FY", "val": 1_000.0},
        {"ticker": "A", "concept": "Revenues", "start": T("2025-01-01"), "end": T("2025-12-31"),
         "filed": T("2026-02-10"), "form": "10-K", "fp": "FY", "val": 500.0},
    ])


def _details():
    asof = {"is": "2025-12-31", "cf": "2025-12-31", "bs": "2025-12-31"}
    return {"A": {"inp": {"totalAssets": 1_005.0, "totalRevenue": 600.0, "capex": -9.0}, "asof": asof,
                  "raw": {"fcf_yield": 0.05}, "pct": {"fcf_yield": 50.0}},
            "B": {"inp": {"totalAssets": 7.0}, "asof": asof, "raw": {}, "pct": {}}}


def test_the_payload_carries_the_check_and_never_scores_it():
    det = _details()
    assert g._attach_filing_check(det, facts=_facts()) == 1
    assert det["A"]["xchk"] == {"ta": pytest.approx(0.005, abs=1e-4), "rev": pytest.approx(100 / 600, abs=1e-4)}
    assert "xchk" not in det["B"]                                      # not in the filings: nothing claimed
    assert "xchk" not in det["A"]["raw"] and "xchk" not in det["A"]["pct"]
    assert set(det["A"]["xchk"]) <= {k for _, k in g._XCHK_FIELDS}     # capex is never shown
    # nothing a scoring module reads mentions it
    for name in ("factor_engine.py", "run_screener.py", "portfolio_constructor.py", "calc_trace.py"):
        src = (ROOT / name).read_text(encoding="utf-8", errors="replace")
        assert "xchk" not in src and "sec_crosscheck" not in src, name


def test_a_build_without_the_cache_omits_the_check_rather_than_failing():
    det = _details()
    assert g._attach_filing_check(det, facts=pd.DataFrame({"nonsense": [1]})) == 0
    assert all("xchk" not in d for d in det.values())


def test_the_check_runs_after_the_scores_are_settled():
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    body = src[src.index("def prepare_dashboard_data"):]
    assert body.index("_reconcile_scores(weights, stock_detail") < body.index("_attach_filing_check(stock_detail)")


def test_the_panel_calls_agreement_at_2_percent():
    html = g.generate_html()
    panel = html[html.index("function renderProvenance(s)"):html.index("// INIT")]
    assert "if (a <= 0.02) agree.push(p[1])" in panel
    assert "Checked against the company&rsquo;s own SEC filing for the same quarters" in panel
    assert "rather than that either figure is wrong" in panel                # the usual reason, stated
    assert "cannot check that they are true" not in panel                    # no longer the whole story
    assert "could not be checked against its filings on this run" in panel   # said when it could not
    for key, name in (("ta", "total assets"), ("eq", "equity"), ("rev", "revenue"), ("ni", "net income"),
                      ("ocf", "operating cash flow")):
        assert f"['{key}', '{name}']" in panel


def test_the_sentences_carry_no_advice():
    from stock_summary import advice_terms_in
    html = g.generate_html()
    panel = html[html.index("function renderProvenance(s)"):html.index("// INIT")]
    assert advice_terms_in(panel) == []
