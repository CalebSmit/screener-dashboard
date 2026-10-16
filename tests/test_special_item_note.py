"""The earnings-yield row says what one-off charge is inside the earnings the run scored.

WHY THIS EXISTS - 2026-10-16, `research/2026-10-15-earnings-definition.md` section 8. A proposal to
score earnings yield on the source's "Normalized Income" - net income without its unusual items -
was **refused**. The research behind it is good (Rouen, So & Wang 2021: next-year persistence 0.810
for core earnings against 0.637 for net income; MSCI's EPS excludes non-recurring items), but at the
four-quarter window the engine sums, 23 companies have no normalized figure at all - Amazon, Bank of
America, Berkshire and Boeing among them - so the metric would rank them on earnings *including*
their one-off items beside 265 companies ranked excluding them. `factor_engine.py:2094` records
Phase 13 fixing that exact defect in this exact metric. And of the 18 fallback companies whose
figures are known, 8 report annual unusual items worth 10% or more of net income (Boeing 434%,
General Mills 2,193%), so the proposal would have removed the small distortions and kept the two
largest in the index.

What shipped instead is this note: the published figure stays, and the row says what is inside it,
from the company's own filings.

Pinned here:

* the charge is the **largest single line the company tagged**, never a sum - the tags overlap (a
  restructuring-and-impairment roll-up contains what the separate impairment tags report), and
  summing them would publish a figure larger than the company reported;
* a **loss is called a loss**, and no share is printed against one. The first draft told readers of
  Gilead, Kraft Heinz, General Mills and Intel that a loss "includes" a charge, and offered "7,887%
  of the net income" against Air Products' $47M loss;
* the recurrence clause counts the years the company **filed**, not the years it charged - counting
  only the latter makes "N of M" report N of N, which was the first draft's second defect;
* it is prose beside the number and reaches no score, no `raw`, no `pct`;
* and the published payload's notes are rebuilt from the cache, so a sentence the filings no longer
  support fails here rather than being published again.
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


def _fact(ticker, concept, val, year=2025, **over):
    return {"ticker": ticker, "concept": concept, "fp": "FY", "val": float(val),
            "start": T(f"{year}-01-01"), "end": T(f"{year}-12-31"),
            "filed": T(f"{year + 1}-02-10"), "form": "10-K", **over}


def _facts():
    """Six filers, built on the shape of the real ones (figures rounded, in dollars):

    PROFIT   - a $1.2B charge inside $2.0B of net income (Starbucks' shape), every year.
    LOSS     - a $7.4B impairment and a $3.4B loss (Kraft Heinz's shape), every year.
    OVERLAP  - files a $500M restructuring-and-impairment roll-up *and* the $300M impairment and
               $400M restructuring lines inside it; the note must use $500M, not $1.2B.
    SMALL    - a $90M charge on $1.0B of net income: 9%, under the threshold.
    REVERSAL - a negative value (a release, not a charge).
    ONCE     - four fiscal years filed, a charge in only the latest one.
    """
    rows = []
    for y in (2022, 2023, 2024, 2025):
        for t, ni in (("PROFIT", 2e9), ("LOSS", -3.4e9), ("OVERLAP", 1e9), ("SMALL", 1e9),
                      ("REVERSAL", 1e9), ("ONCE", 1e9)):
            rows.append(_fact(t, "NetIncomeLoss", ni, y))          # the year anchor
        rows.append(_fact("PROFIT", "RestructuringSettlementAndImpairmentProvisions", 1.2e9, y))
        rows.append(_fact("LOSS", "GoodwillAndIntangibleAssetImpairment", 7.4e9, y))
        rows.append(_fact("OVERLAP", "RestructuringCostsAndAssetImpairmentCharges", 0.5e9, y))
        rows.append(_fact("OVERLAP", "AssetImpairmentCharges", 0.3e9, y))
        rows.append(_fact("OVERLAP", "RestructuringCharges", 0.4e9, y))
        rows.append(_fact("SMALL", "RestructuringCharges", 0.09e9, y))
        rows.append(_fact("REVERSAL", "RestructuringCharges", -0.5e9, y))
    rows.append(_fact("ONCE", "RestructuringCharges", 0.4e9, 2025))
    return pd.DataFrame(rows)


def _details():
    ni = {"PROFIT": 2e9, "LOSS": -3.4e9, "OVERLAP": 1e9, "SMALL": 1e9,
          "REVERSAL": 1e9, "ONCE": 1e9}
    return {t: {"inp": {"netIncome": v, "marketCap": 40e9},
                "asof": {"is": "2025-12-31", "cf": "2025-12-31", "bs": "2025-12-31"},
                "raw": {"earnings_yield": v / 40e9}, "pct": {"earnings_yield": 50.0}}
            for t, v in ni.items()}


def _notes():
    det = _details()
    n = g._attach_special_item_note(det, facts=_facts())
    return det, n


# ------------------------------------------------------------------ what the sentence says
def test_the_note_states_the_filed_charge_and_its_tag():
    det, n = _notes()
    assert n == 4, "PROFIT, LOSS, OVERLAP and ONCE qualify; SMALL and REVERSAL do not"
    note = det["PROFIT"]["cavs"]["earnings_yield"]
    assert "$2.0B of net income above is after at least $1.2B" in note
    assert "restructuring, settlement and impairment provisions" in note
    assert "RestructuringSettlementAndImpairmentProvisions" in note
    assert "equal to 60% of that net income, before any tax effect" in note
    assert "computed from earnings as reported, so the charge is in it" in note


def test_a_loss_is_called_a_loss_and_carries_no_share():
    """The first draft printed `_usd`, which takes an absolute value, so a loss read as a profit -
    and a percentage of a loss is arithmetic, not information."""
    det, _ = _notes()
    note = det["LOSS"]["cavs"]["earnings_yield"]
    assert "$3.4B loss above is after at least $7.4B" in note
    assert "of net income above" not in note
    assert "% of that net income" not in note and "times that net income" not in note
    # the profit case still shows its share, so the omission is about the sign and nothing else
    assert "% of that net income" in det["PROFIT"]["cavs"]["earnings_yield"]


def test_the_largest_single_tag_is_used_never_a_sum():
    """OVERLAP files a roll-up and the two lines inside it. $500M is the company's figure; $1.2B
    is three tags added together and larger than anything it reported."""
    det, _ = _notes()
    note = det["OVERLAP"]["cavs"]["earnings_yield"]
    assert "at least $500M" in note
    assert "$1.2B" not in note and "$900M" not in note
    assert "RestructuringCostsAndAssetImpairmentCharges" in note
    entry = X.special_items({"stock_detail": _details()}, _facts())["OVERLAP"]
    assert entry["charge"] == pytest.approx(0.5e9)


def test_a_charge_below_the_threshold_gets_no_note():
    det, _ = _notes()
    assert "cavs" not in det["SMALL"] and "cavs" not in det["REVERSAL"]
    book = X.special_items({"stock_detail": _details()}, _facts())
    assert set(book) == {"PROFIT", "LOSS", "OVERLAP", "ONCE"}


def test_the_threshold_is_the_registered_ten_percent():
    assert X.SPECIAL_ITEM_SHARE_MIN == 0.10
    facts = _facts()
    for val, expected in ((0.10e9, True), (0.099e9, False)):
        f = facts.copy()
        f.loc[(f.ticker == "SMALL") & (f.concept == "RestructuringCharges"), "val"] = val
        assert ("SMALL" in X.special_items({"stock_detail": _details()}, f)) is expected


def test_the_recurrence_clause_counts_the_filed_years():
    """Counting only the years that carry a charge makes "N of M" report N of N. ONCE files four
    fiscal years and charged in one of them."""
    det, _ = _notes()
    assert "in 1 of its last 4 fiscal years" in det["ONCE"]["cavs"]["earnings_yield"]
    assert "in 4 of its last 4 fiscal years" in det["PROFIT"]["cavs"]["earnings_yield"]
    facts = _facts()
    assert X._charge_years(facts[facts.ticker == "ONCE"], T("2025-12-31")) == (1, 4)
    assert X._charge_years(facts[facts.ticker == "PROFIT"], T("2025-12-31")) == (4, 4)
    # a company with no annual anchor gets no clause rather than a wrong one
    bare = facts[(facts.ticker == "ONCE") & (facts.concept == "RestructuringCharges")]
    assert X._charge_years(bare, T("2025-12-31")) == (0, 0)


def test_the_charge_is_rebuilt_for_the_period_the_run_scored():
    """The two figures in the sentence have to describe one period, so the charge is taken at the
    income statement's own quarter - the same rule as the filing check."""
    det = _details()
    det["PROFIT"]["asof"]["is"] = "2019-12-31"      # before anything in the cache
    assert "PROFIT" not in X.special_items({"stock_detail": det}, _facts())


# ------------------------------------------------------------------ it stays display-only
def test_the_note_never_reaches_a_score():
    det, _ = _notes()
    for d in det.values():
        assert "cavs" not in d.get("raw", {}) and "cavs" not in d.get("pct", {})
        assert set(d["raw"]) == {"earnings_yield"} and d["pct"] == {"earnings_yield": 50.0}
    for name in ("factor_engine.py", "run_screener.py", "portfolio_constructor.py", "calc_trace.py",
                 "improvement_engine.py", "backtest.py"):
        src = (ROOT / name).read_text(encoding="utf-8", errors="replace")
        assert "special_items" not in src and "special_item_note" not in src, name
        assert not re.search(r"^\s*(import sec_crosscheck|from sec_crosscheck)\b", src, flags=re.M), name


def test_the_sentence_carries_no_advice():
    from stock_summary import advice_terms_in
    det, _ = _notes()
    for d in det.values():
        assert advice_terms_in((d.get("cavs") or {}).get("earnings_yield", "")) == []


def test_a_build_without_the_cache_says_nothing_rather_than_failing():
    det = _details()
    assert g._attach_special_item_note(det, facts=pd.DataFrame({"nonsense": [1]})) == 0
    assert all("cavs" not in d for d in det.values())


def test_a_long_filing_tag_can_wrap():
    """The note names the tag so a reader can find the figure in the filing, and the longest is
    `ImpairmentOfIntangibleAssetsIndefinitelivedExcludingGoodwill` - 59 characters, one unbreakable
    token. Without `overflow-wrap: anywhere` it is clipped at the panel's edge at 375px (seen on
    Gilead, rank 53, 2026-10-16). The clip does not widen the document, so a `scrollWidth` check
    cannot see it - which is why this is pinned on the rule rather than on a viewport."""
    longest = max(X.SPECIAL_ITEM_CONCEPTS, key=len)
    assert len(longest) >= 55, longest
    html = g.generate_html()
    assert re.search(r"\.wk-caveat-stock\s*\{[^}]*overflow-wrap:\s*anywhere", html), \
        "the stock-note block must break a long tag rather than clip it"
    assert "In this company&rsquo;s filings." in html


def test_it_does_not_overwrite_the_capex_note():
    """Both notes live in `cavs`, keyed by metric. One must not clear the other."""
    det = _details()
    det["PROFIT"]["cavs"] = {"fcf_yield": "the capex sentence"}
    g._attach_special_item_note(det, facts=_facts())
    assert det["PROFIT"]["cavs"]["fcf_yield"] == "the capex sentence"
    assert "earnings_yield" in det["PROFIT"]["cavs"]


# ------------------------------------------------------------------ against the real run
def test_the_published_payload_agrees_with_the_filings():
    """The committed payload's notes are rebuilt from the cache and the published inputs, so a
    sentence the filings no longer support fails here rather than being published again."""
    pytest.importorskip("pyarrow")
    from sec_fundamentals import FACTS_PATH
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
    published = {t: (d.get("cavs") or {}).get("earnings_yield")
                 for t, d in payload["stock_detail"].items()
                 if (d.get("cavs") or {}).get("earnings_yield")}
    if not FACTS_PATH.exists():
        pytest.skip("no SEC cache on this machine")
    found = X.special_items(payload, pd.read_parquet(FACTS_PATH))
    rebuilt = {t: X.special_item_note(e) for t, e in found.items()}
    if not published:
        # the payload predates the note; the rebuild still has to be sane where it fires
        for t, e in found.items():
            assert e["charge"] > 0 and e["share"] >= X.SPECIAL_ITEM_SHARE_MIN, t
            assert e["of_years"] >= e["charge_years"], t
        pytest.skip("published payload has no notes yet (pre-dates 2026-10-16)")
    assert published == rebuilt
