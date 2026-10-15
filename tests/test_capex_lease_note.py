"""The FCF-yield row says what the capex figure contains, for a captive finance arm's lease fleet.

WHY THIS EXISTS - 2026-10-15, `research/2026-10-14-capex-definition.md` section 5. Free cash flow
is operating cash flow minus Yahoo's `Capital Expenditure` row, and for four S&P 500 companies most
of that row is the vehicles or equipment a captive finance arm buys to lease to customers: 61% of
GM's $22.6B. A proposal to subtract those purchases was **refused** - the ratio would then drop the
finance arm's capex while keeping its operating cash flow (GM's own 10-K: automotive operating cash
flow $18.7B against the consolidated $26.87B) and its debt in the enterprise value. What shipped
instead is this note, which decomposes the published figure from the company's own filings.

Pinned here:

* the note's arithmetic is the **company's**, and it has to reconcile with the capex the run used -
  without that check the row would have told readers that 577% of Ford's capex was vehicles for
  lease, because Yahoo's Ford capex already excludes Ford Credit;
* the proceeds leg is claimed only for a company that files it (of the four, only GM);
* it is prose beside the number and reaches no score, no `raw`, no `pct`;
* and the published payload's notes still agree with the filings, so a build cannot publish a
  sentence the cache no longer supports.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as g  # noqa: E402
import sec_crosscheck as X  # noqa: E402

T = pd.Timestamp
YEAR = {"start": T("2025-01-01"), "end": T("2025-12-31"), "filed": T("2026-02-10"), "form": "10-K"}


def _fact(ticker, concept, val, **over):
    return {"ticker": ticker, "concept": concept, "fp": "FY", "val": float(val), **YEAR, **over}


def _facts():
    """Four filers, built on the shape of the real ones (figures rounded, in dollars):

    CAPTIVE - lease purchases are 60% of a capex figure they reconcile with, and it files the
              proceeds leg (GM).
    NOPROC  - the same, with no proceeds tag filed (DE, CAT, PCAR).
    OUTSIDE - files a far larger loans-and-leases line that is **not** inside its capex (Ford).
    SMALL   - files a lease line worth 5% of capex, under the threshold.
    """
    return pd.DataFrame([
        _fact("CAPTIVE", "PaymentsToAcquirePropertyPlantAndEquipment", 4e9),
        _fact("CAPTIVE", "PaymentsToAcquireLeasesHeldForInvestment", 6e9),
        _fact("CAPTIVE", "ProceedsFromLeasesHeldForInvestment", 4.5e9),
        _fact("NOPROC", "PaymentsToAcquirePropertyPlantAndEquipment", 1.2e9),
        _fact("NOPROC", "PaymentsToAcquireEquipmentOnLease", 2.8e9),
        _fact("OUTSIDE", "PaymentsToAcquirePropertyPlantAndEquipment", 9.6e9),
        _fact("OUTSIDE", "PaymentsToAcquireLoansAndLeasesHeldForInvestment", 55.7e9),
        _fact("SMALL", "PaymentsToAcquirePropertyPlantAndEquipment", 9.5e9),
        _fact("SMALL", "PaymentsToAcquireEquipmentOnLease", 0.5e9),
    ])


def _details():
    asof = {"is": "2025-12-31", "cf": "2025-12-31", "bs": "2025-12-31"}
    cap = {"CAPTIVE": -10e9, "NOPROC": -4e9, "OUTSIDE": -9.6e9, "SMALL": -10e9}
    return {t: {"inp": {"capex": v, "operatingCashFlow": 20e9}, "asof": dict(asof),
                "raw": {"fcf_yield": 0.05}, "pct": {"fcf_yield": 50.0}} for t, v in cap.items()}


# --------------------------------------------------------------------------- the decomposition
def test_the_note_decomposes_the_capex_the_run_used():
    det = _details()
    assert g._attach_capex_lease_note(det, facts=_facts()) == 2
    note = det["CAPTIVE"]["cavs"]["fcf_yield"]
    # every figure in the sentence is a filed one, and the two parts add to the capex above
    assert "$10.0B of capital expenditure" in note
    assert "$4.0B of property, plant and equipment" in note
    assert "$6.0B (60% of it)" in note
    assert "PaymentsToAcquireLeasesHeldForInvestment" in note
    entry = X.lease_book({"stock_detail": _details()}, _facts())["CAPTIVE"]
    assert entry["ppe"] + entry["purchases"] == pytest.approx(entry["capex"], rel=X.LEASE_RECONCILE_TOL)
    # and it says whose measure this is not
    assert "industrial business only" in note and "not the company's own measure" in note


def test_a_company_whose_lease_line_is_not_inside_capex_gets_no_note():
    """Ford's case, and the reason the reconciliation exists: it files $55.7B of loan and lease
    purchases against $9.6B of capex, because Yahoo's figure already excludes Ford Credit."""
    det = _details()
    g._attach_capex_lease_note(det, facts=_facts())
    assert "cavs" not in det["OUTSIDE"]
    book = X.lease_book({"stock_detail": _details()}, _facts())
    assert "OUTSIDE" not in book and "SMALL" not in book        # outside, and under the threshold
    assert set(book) == {"CAPTIVE", "NOPROC"}


def test_the_threshold_is_the_registered_ten_percent():
    assert X.LEASE_SHARE_MIN == 0.10
    facts = _facts()
    # lift SMALL's lease line to 10% of its capex and it qualifies; at 9.9% it does not
    for val, expected in ((1.0e9, True), (0.99e9, False)):
        f = facts.copy()
        f.loc[(f.ticker == "SMALL") & (f.concept == "PaymentsToAcquireEquipmentOnLease"), "val"] = val
        f.loc[(f.ticker == "SMALL") & (f.concept == "PaymentsToAcquirePropertyPlantAndEquipment"), "val"] = 10e9 - val
        assert ("SMALL" in X.lease_book({"stock_detail": _details()}, f)) is expected


def test_the_proceeds_sentence_appears_only_when_the_company_files_them():
    det = _details()
    g._attach_capex_lease_note(det, facts=_facts())
    with_proceeds = det["CAPTIVE"]["cavs"]["fcf_yield"]
    without = det["NOPROC"]["cavs"]["fcf_yield"]
    assert "$4.5B received when those assets are sold" in with_proceeds
    assert "ProceedsFromLeasesHeldForInvestment" in with_proceeds
    assert "received when those assets are sold" not in without
    assert "sold at lease end" not in without


# --------------------------------------------------------------------------- it stays display-only
def test_the_note_never_reaches_a_score():
    det = _details()
    g._attach_capex_lease_note(det, facts=_facts())
    for t, d in det.items():
        assert "cavs" not in d.get("raw", {}) and "cavs" not in d.get("pct", {})
        assert d["raw"] == {"fcf_yield": 0.05} and d["pct"] == {"fcf_yield": 50.0}
    for name in ("factor_engine.py", "run_screener.py", "portfolio_constructor.py", "calc_trace.py",
                 "improvement_engine.py", "backtest.py"):
        src = (ROOT / name).read_text(encoding="utf-8", errors="replace")
        assert "lease_book" not in src and "lease_capex_note" not in src, name
        assert not re.search(r"^\s*(import sec_crosscheck|from sec_crosscheck)\b", src, flags=re.M), name


def test_the_sentence_carries_no_advice():
    from stock_summary import advice_terms_in
    det = _details()
    g._attach_capex_lease_note(det, facts=_facts())
    for d in det.values():
        assert advice_terms_in((d.get("cavs") or {}).get("fcf_yield", "")) == []


def test_a_build_without_the_cache_says_nothing_rather_than_failing():
    det = _details()
    assert g._attach_capex_lease_note(det, facts=pd.DataFrame({"nonsense": [1]})) == 0
    assert all("cavs" not in d for d in det.values())


def test_the_page_renders_a_per_stock_note_under_the_metric():
    html = g.generate_html()
    assert "const stockCav = (s.cavs || {})[m];" in html
    assert "In this company&rsquo;s filings." in html
    assert ".wk-caveat-stock" in html


# --------------------------------------------------------------------------- against the real run
def test_the_published_payload_agrees_with_the_filings():
    """The committed payload's notes are rebuilt from the cache and the published inputs, so a
    sentence the filings no longer support fails here rather than being published again."""
    pytest.importorskip("pyarrow")
    from sec_fundamentals import FACTS_PATH
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
    published = {t: (d.get("cavs") or {}).get("fcf_yield")
                 for t, d in payload["stock_detail"].items() if (d.get("cavs") or {}).get("fcf_yield")}
    if not FACTS_PATH.exists():
        pytest.skip("no SEC cache on this machine")
    book = X.lease_book(payload, pd.read_parquet(FACTS_PATH))
    rebuilt = {t: X.lease_capex_note(e) for t, e in book.items()}
    if not published:
        # the payload predates the note; the rebuild still has to be sane where it fires
        for t, e in book.items():
            assert e["ppe"] + e["purchases"] == pytest.approx(e["capex"], rel=X.LEASE_RECONCILE_TOL), t
        pytest.skip("published payload has no notes yet (pre-dates 2026-10-15)")
    assert published == rebuilt
