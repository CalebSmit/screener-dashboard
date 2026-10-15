"""A stock scored a period behind its own filings says so (``sec_crosscheck.newer_filings``).

2026-10-15: 32 of 501 stocks were scored on statements a quarter older than the newest period the
company had filed; for 16 the filing was over 70 days old (Amazon, Bank of America, Boeing - the
source listed the newer quarter as an empty column). The engine is right not to read an empty
column. What is pinned here: the stock is named, only filings made by the run date count, a date
wobble is not a newer period, the drilldown says the figures are the period before, the morning
health note counts the old ones, and nothing of it reaches a score.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as g  # noqa: E402
import sec_crosscheck as X  # noqa: E402

T = pd.Timestamp


def _facts():
    return pd.DataFrame([
        # AMZN-like: the run used Q1; Q2 was filed on 2026-07-31
        {"ticker": "A", "concept": "Revenues", "start": T("2026-01-01"), "end": T("2026-03-31"),
         "filed": T("2026-05-01"), "form": "10-Q", "fp": "Q1", "val": 1.0},
        {"ticker": "A", "concept": "Revenues", "start": T("2026-04-01"), "end": T("2026-06-30"),
         "filed": T("2026-07-31"), "form": "10-Q", "fp": "Q2", "val": 1.0},
        # up to date
        {"ticker": "B", "concept": "Revenues", "start": T("2026-04-01"), "end": T("2026-06-30"),
         "filed": T("2026-07-29"), "form": "10-Q", "fp": "Q2", "val": 1.0},
        # a 52/53-week calendar: a few days' difference is the same period
        {"ticker": "C", "concept": "Revenues", "start": T("2026-03-30"), "end": T("2026-06-28"),
         "filed": T("2026-07-30"), "form": "10-Q", "fp": "Q2", "val": 1.0},
        # filed after the run: invisible to it
        {"ticker": "D", "concept": "Revenues", "start": T("2026-07-01"), "end": T("2026-09-30"),
         "filed": T("2026-10-20"), "form": "10-Q", "fp": "Q3", "val": 1.0},
    ])


def _payload():
    used = lambda d: {"inp": {}, "asof": {"is": d, "bs": d, "cf": d}, "raw": {}, "pct": {}}  # noqa: E731
    return {"kpis": {"run_timestamp": "2026-10-15T02:00:00"},
            "stock_detail": {"A": used("2026-03-31"), "B": used("2026-06-30"), "C": used("2026-06-30"),
                             "D": used("2026-06-30")}}


def test_a_stock_behind_its_filings_is_named():
    out = X.newer_filings(_payload(), _facts())
    assert out == {"A": {"end": "2026-06-30", "filed": "2026-07-31", "form": "10-Q"}}


def test_only_filings_made_by_the_run_date_count():
    assert "D" not in X.newer_filings(_payload(), _facts())
    later = X.newer_filings(_payload(), _facts(), today="2026-10-21")
    assert later["D"]["end"] == "2026-09-30"


def test_the_payload_carries_it_and_never_scores_it():
    det = _payload()["stock_detail"]
    g._attach_filing_check(det, facts=_facts(), run_date="2026-10-15T02:00:00")
    assert det["A"]["newer_filing"]["filed"] == "2026-07-31"
    assert "newer_filing" not in det["B"]
    assert "newer_filing" not in det["A"]["raw"] and "newer_filing" not in det["A"]["pct"]
    for name in ("factor_engine.py", "calc_trace.py", "portfolio_constructor.py"):
        assert "newer_filing" not in (ROOT / name).read_text(encoding="utf-8", errors="replace"), name


def test_the_panel_says_the_figures_are_the_period_before():
    html = g.generate_html()
    panel = html[html.index("function renderProvenance(s)"):html.index("// INIT")]
    assert "A period behind the filings." in panel
    assert "so the figures here are the period before" in panel
    assert "escapeHtml(nf.form)" in panel


def test_the_health_note_counts_the_old_ones():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_run_health as h
    src = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
    assert "stocks scored a period behind a filing over 30 days old" in src
    assert re.search(r"X\.newer_filings\(payload, facts\)", src)
    assert h.check_inputs_against_filings  # the note lives in the filing check's note line
