"""The rows that read operating cash flow say when the source counted part of a year twice.

WHY THIS EXISTS - 2026-11-17, `research/2026-11-17-ytd-in-a-quarterly-column.md`. A 10-Q's
cash-flow statement is cumulative and the data source derives each quarter by subtraction. For
DexCom's September 2026 quarter it did not: the "quarterly" column carried $1.120bn, the nine
months as filed, so the four quarters the engine sums came to $2.209bn against $1.414bn filed -
on the stock ranked 8th, whose free-cash-flow yield sat at the 84th percentile of its sector on
that figure. The page's only sentence about it said a difference "usually means the two sources
define the line differently ... rather than that either figure is wrong".

Measured the same day: 3 of 2,300 quarterly columns are year-to-date figures, 1 inside the
twelve months scored; nothing the source returns can recognise one (the best statement-only rule
raises 62 false alarms for the one true case); three filed figures can, exactly, and over every
published page since 2026-10-08 that identity fires on DexCom's 8 pages and on none of the 143
other stock-days more than 2% from a filing.

Pinned here:

* the identity itself - the run exceeds the filed twelve months by the filed year-to-date figure
  one quarter before one of the last **four** quarter ends - on a six-month, a nine-month and a
  fiscal-year column, with a negative cash flow, and **not** on a disagreement of the same size
  that is not that amount;
* both of its bars, as literals and on both sides (a tolerance read from the constant under test
  cannot see the constant move - `plan/claims-mutation-audit.md` row 54);
* every figure in the sentence is a filed one or the run's own, and it says the score was not
  adjusted;
* it has its **own key** (`srcn`), because the capex and special-item notes' publish-gate tests
  compare the whole of `cavs` with what the filings rebuild;
* it is prose beside the number and reaches no score - and the published payload's notes are
  rebuilt from the cache, so a sentence the filings no longer support fails here.
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
OCF = "NetCashProvidedByUsedInOperatingActivities"


def _f(ticker, start, end, val, form="10-Q"):
    return {"ticker": ticker, "concept": OCF, "fp": "", "val": float(val), "start": T(start),
            "end": T(end), "filed": T(end) + pd.Timedelta(days=30), "form": form}


def _filer(ticker, scale=1.0):
    """DexCom's filed operating cash flow, in $M, as its 10-Ks and 10-Qs carry it: cumulative
    from the start of the fiscal year. The quarters these imply for 2026 are 525.6, 269.2 and
    325.6; the fourth quarter of 2025 is 294.0; the twelve months to 2026-09-30 are 1,414.4."""
    m = 1e6 * scale
    return [
        _f(ticker, "2024-01-01", "2024-12-31", 989.5 * m, "10-K"),
        _f(ticker, "2025-01-01", "2025-03-31", 183.8 * m),
        _f(ticker, "2025-01-01", "2025-06-30", 486.8 * m),
        _f(ticker, "2025-01-01", "2025-09-30", 1146.7 * m),
        _f(ticker, "2025-01-01", "2025-12-31", 1440.7 * m, "10-K"),
        _f(ticker, "2026-01-01", "2026-03-31", 525.6 * m),
        _f(ticker, "2026-01-01", "2026-06-30", 794.8 * m),
        _f(ticker, "2026-01-01", "2026-09-30", 1120.4 * m),
    ]


TTM = 1414.4e6          # 1120.4 + 1440.7 - 1146.7, the filed twelve months to 2026-09-30
RUNS = {
    "NINE": 2209.2e6,       # 294.0 + 525.6 + 269.2 + 1120.4: the September column is nine months
    "RIGHT": TTM,           # the four filed quarters
    "SIX": TTM + 525.6e6,   # the June column carries 794.8, six months: the first quarter twice
    "OTHER": TTM + 600e6,   # as far from the filing as NINE, and not by a year-to-date figure
}


def _facts():
    rows = []
    for t in RUNS:
        rows += _filer(t)
    rows += _filer("LOSS", scale=-1.0)
    return pd.DataFrame(rows)


def _details(runs=None, asof="2026-09-30"):
    runs = dict(RUNS, LOSS=-2209.2e6) if runs is None else runs
    return {t: {"inp": {"operatingCashFlow": v, "capex": -477.4e6, "netIncome": 991e6, "totalAssets": 6238.8e6},
                "asof": {"is": asof, "cf": asof, "bs": asof},
                "raw": {"fcf_yield": 0.0577, "accruals": -0.1953},
                "pct": {"fcf_yield": 83.6, "accruals": 97.5}} for t, v in runs.items()}


# ------------------------------------------------------------------ the identity
def test_a_nine_month_figure_in_the_september_column_is_recognised():
    found = X.ytd_columns({"stock_detail": _details()}, _facts())
    e = found["NINE"]
    assert e["run"] == pytest.approx(2209.2e6) and e["filed"] == pytest.approx(1414.4e6)
    assert e["column_end"] == "2026-09-30" and e["months"] == 9
    assert e["column_ytd"] == pytest.approx(1120.4e6)
    assert e["column_quarter"] == pytest.approx(325.6e6)          # 1120.4 - 794.8, as filed
    assert e["rel"] == pytest.approx((2209.2 - 1414.4) / 2209.2)


def test_only_the_stocks_whose_gap_is_a_year_to_date_figure_are_named():
    """RIGHT agrees with its filing. OTHER is 30% from it - further than SIX - by an amount no
    quarter end's year-to-date figure explains, and a note that fired on it would be a guess."""
    found = X.ytd_columns({"stock_detail": _details()}, _facts())
    assert set(found) == {"NINE", "SIX", "LOSS"}


def test_an_earlier_column_inside_the_twelve_months_is_found_and_named():
    e = X.ytd_columns({"stock_detail": _details()}, _facts())["SIX"]
    assert e["column_end"] == "2026-06-30" and e["months"] == 6
    assert e["column_ytd"] == pytest.approx(794.8e6) and e["column_quarter"] == pytest.approx(269.2e6)
    assert "the six months to that date" in X.ytd_column_note(e)
    assert "quarter to June 30, 2026" in X.ytd_column_note(e)


def test_a_column_outside_the_twelve_months_is_not_looked_at():
    """The engine sums four quarters. The September 2025 column is the fifth back from
    September 2026; a year-to-date figure there (DexCom's also carries one) is in no twelve
    months the run scored, and the gap it would explain - 486.8, the six months to June 2025 -
    must not be matched."""
    det = _details({"FIFTH": TTM + 486.8e6})
    facts = pd.DataFrame(_filer("FIFTH"))
    assert X.ytd_columns({"stock_detail": det}, facts) == {}


def test_a_fiscal_year_in_the_fourth_quarter_column():
    """Statement dated at the fiscal year end: the filed twelve months are the 10-K's figure, and
    a December column carrying the whole year adds the nine months to September."""
    det = _details({"YEAR": 1440.7e6 + 1146.7e6}, asof="2025-12-31")
    e = X.ytd_columns({"stock_detail": det}, pd.DataFrame(_filer("YEAR")))["YEAR"]
    assert e["filed"] == pytest.approx(1440.7e6) and e["months"] == 12
    assert e["column_quarter"] == pytest.approx(294.0e6)
    assert "the twelve months to that date" in X.ytd_column_note(e)


def test_the_match_is_within_half_a_percent_of_the_runs_figure():
    assert X.YTD_TOL == 0.005
    facts = pd.DataFrame(_filer("NEAR"))
    run = 2209.2e6
    for off, expected in ((0.004, True), (-0.004, True), (0.006, False), (-0.006, False)):
        det = _details({"NEAR": run + off * run})
        assert ("NEAR" in X.ytd_columns({"stock_detail": det}, facts)) is expected, off


def test_an_excess_under_two_percent_of_the_figure_gets_no_note():
    """A first quarter near zero makes the six-month figure and the second quarter almost the
    same number; a note about a 1.9% difference is noise, and the filing check's own bar is 2%."""
    assert X.YTD_MIN_SHARE == 0.02
    # the run is 888.8 + 2 x the first quarter here, so 19.5 is 2.10% of it and 17.5 is 1.89%
    for q1, expected in ((19.5, True), (17.5, False)):
        rows = [r for r in _filer("TINY") if r["end"] > T("2025-12-31") or r["form"] == "10-K"]
        rows += [_f("TINY", "2025-01-01", "2025-09-30", 1146.7e6)]
        for r in rows:
            if r["start"] == T("2026-01-01"):
                r["val"] += (q1 - 525.6) * 1e6                  # move the first quarter, keep the others
        facts = pd.DataFrame(rows)
        ttm, _ = X.flow_ttm(facts, T("2026-09-30"))
        det = _details({"TINY": ttm + q1 * 1e6})
        assert (q1 * 1e6 / (ttm + q1 * 1e6) >= 0.02) is expected, "the fixture brackets the literal bar"
        assert ("TINY" in X.ytd_columns({"stock_detail": det}, facts)) is expected, q1


# ------------------------------------------------------------------ what the sentence says
def test_the_note_states_the_runs_figure_and_the_three_filed_ones():
    det = _details()
    assert g._attach_ytd_column_note(det, facts=_facts()) == 3
    note = det["NINE"]["srcn"]["fcf_yield"]
    assert note == det["NINE"]["srcn"]["accruals"], "one note, on both rows that read the figure"
    assert set(det["NINE"]["srcn"]) == set(X.YTD_METRICS) == {"fcf_yield", "accruals"}
    assert "The $2.2B of operating cash flow above counts part of the year twice." in note
    assert "column for the quarter to September 30, 2026 carries $1.1B" in note
    assert "this company's filed figure for the nine months to that date" in note
    assert "the quarter alone was $326M as filed" in note
    assert "the twelve months to September 30, 2026 come to $1.4B in the company's filings" in note
    assert "36% below the figure used here" in note
    assert "computed from the source's figure as delivered and has not been adjusted" in note
    assert det["NINE"]["xwhy"] == {"ocf": "ytd"}
    assert "srcn" not in det["RIGHT"] and "srcn" not in det["OTHER"]
    assert "xwhy" not in det["RIGHT"] and "xwhy" not in det["OTHER"]


def test_a_cash_outflow_keeps_its_sign():
    """`_usd` prints a magnitude; the special-item note's first draft called a loss a profit that
    way. LOSS files the mirror image of DexCom's figures."""
    det = _details()
    g._attach_ytd_column_note(det, facts=_facts())
    note = det["LOSS"]["srcn"]["accruals"]
    assert "The -$2.2B of operating cash flow" in note
    assert "carries -$1.1B" in note and "the quarter alone was -$326M" in note
    assert "come to -$1.4B" in note
    assert "36% above the figure used here" in note, "the filed outflow is the smaller one"


def test_the_sentence_carries_no_advice():
    from stock_summary import advice_terms_in
    det = _details()
    g._attach_ytd_column_note(det, facts=_facts())
    for d in det.values():
        for note in (d.get("srcn") or {}).values():
            assert advice_terms_in(note) == []


# ------------------------------------------------------------------ it stays display-only
def test_the_note_never_reaches_a_score_and_has_its_own_key():
    det = _details()
    det["NINE"]["cavs"] = {"fcf_yield": "the capex sentence"}
    g._attach_ytd_column_note(det, facts=_facts())
    assert det["NINE"]["cavs"] == {"fcf_yield": "the capex sentence"}, "the lease note's key is not shared"
    for d in det.values():
        assert d["raw"] == {"fcf_yield": 0.0577, "accruals": -0.1953}
        assert d["pct"] == {"fcf_yield": 83.6, "accruals": 97.5}
        assert d["inp"]["operatingCashFlow"] in set(RUNS.values()) | {-2209.2e6}, "the input is as delivered"
    for name in ("factor_engine.py", "run_screener.py", "portfolio_constructor.py", "calc_trace.py",
                 "improvement_engine.py", "backtest.py"):
        src = (ROOT / name).read_text(encoding="utf-8", errors="replace")
        assert "ytd_columns" not in src and "ytd_column_note" not in src and "srcn" not in src, name
        assert not re.search(r"^\s*(import sec_crosscheck|from sec_crosscheck)\b", src, flags=re.M), name


def test_a_build_without_the_cache_says_nothing_rather_than_failing():
    det = _details()
    assert g._attach_ytd_column_note(det, facts=pd.DataFrame({"nonsense": [1]})) == 0
    assert all("srcn" not in d and "xwhy" not in d for d in det.values())


def test_the_page_renders_the_note_under_the_metric_and_in_the_provenance_panel():
    html = g.generate_html()
    assert "const srcNote = (s.srcn || {})[m];" in html
    assert "Against this company&rsquo;s filings." in html
    panel = html[html.index("function renderProvenance(s)"):html.index("// INIT")]
    assert "(s.xwhy || {}).ocf === 'ytd'" in panel
    assert "Operating cash flow counts part of the year twice." in panel
    # the "usually a definition" sentence is kept for the *other* fields that differ, and only them
    assert "if (differ.length > (ytdCol ? 1 : 0))" in panel
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert '"src_notes": _n_ytd,' in src
    assert src.index("_attach_filing_check(stock_detail, run_date") < src.index("_n_ytd = _attach_ytd_column_note(stock_detail)")


# ------------------------------------------------------------------ against the real run
def test_the_published_payload_agrees_with_the_filings():
    """The committed payload's notes are rebuilt from the cache and the published inputs. The
    premise is asserted, not assumed: only a payload that says it was built with the note
    (``src_notes``) and from the cache on disk is compared - which stocks qualify is the day's
    data and is never named here."""
    pytest.importorskip("pyarrow")
    from sec_fundamentals import FACTS_PATH
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
    # `inp` (the published operating cash flow ytd_columns reads) is in dashboard_detail.js since
    # 2026-11-22 - fold the shards back as the publish gate does.
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_run_health as _health
    payload = _health.fold_shards(payload, _health._dir_reader(ROOT))
    if "src_notes" not in payload:
        pytest.skip("published payload was built before the note existed (2026-11-17)")
    if not FACTS_PATH.exists():
        pytest.skip("no SEC cache on this machine")
    facts = pd.read_parquet(FACTS_PATH)
    built_from, on_disk = payload.get("sec_facts"), g._facts_vintage(facts)
    if built_from != on_disk:
        pytest.skip(f"MISSING DATA: the payload was built from SEC cache {built_from}, "
                    f"the cache on disk is {on_disk} - not the filings these notes came from")
    published = {t: d["srcn"] for t, d in payload["stock_detail"].items() if d.get("srcn")}
    found = X.ytd_columns(payload, facts)
    rebuilt = {t: {m: X.ytd_column_note(e) for m in X.YTD_METRICS} for t, e in found.items()}
    assert published == rebuilt
    assert payload["src_notes"] == len(found)
    assert {t for t, d in payload["stock_detail"].items() if (d.get("xwhy") or {}).get("ocf") == "ytd"} == set(found)
    for t, e in found.items():      # the filing check on the same page must show the same gap
        assert abs(payload["stock_detail"][t]["xchk"]["ocf"]) == pytest.approx(abs(e["rel"]), abs=1e-4), t
