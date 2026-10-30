"""The eleven badges that lived in dead code, and the four that said nothing true.

WHY THIS EXISTS - 2026-10-27. On 2026-10-15 a session found that
``generate_dashboard.renderFlagsWarnings`` was dead: nothing called it, and its container
``#modal-flags`` was never written into the page. That morning's quarterly-rejection badge had
been put inside it and could never have appeared. Its two provenance sentences were rehomed the
same day; the other **eleven** badges were left, with the note "one session should check each
against the live page and either rehome it or delete it" - which is this one, twelve days later.

Measured on the 2026-10-27 published run (498 stocks,
``research/measurements/2026-10-27-dead-badge-audit.py``). Seven of the eleven reach the reader
by another route. Four did not:

* **LTM 3-of-4 annualised - 4 stocks (CCL, EFX, TRV at rank 22, V), and a false sentence.**
  The provenance panel picks its lead on ``fl.ltm_rolled || fl.q_rejected``. All four of these
  have both false and ``data_source == "quarterly"``, so the live page told readers their
  "income and cash-flow figures are the sum of the last four reported quarters" while the engine
  had summed three and scaled by 4/3. This is the same defect the 2026-10-15 session fixed for
  the other two constructions, still live for the third.
* **Enterprise value substituted - 18 stocks.** The rule has been on the methodology page for
  a long time ("discrepancies > 10% (25% for Financials) trigger automatic correction"), but no
  stock's page said it had fired for *that stock*, so a reader checking the enterprise value in
  the workings against the provider's own figure would find two numbers and no explanation.
  It feeds three scored metrics: EV/EBITDA, EV/Sales and free-cash-flow yield.
* **Beta overlap under 80% - 1 stock (FDXF, 42%).** Beta is then not computed at all, so the
  metric row said "no data" with no reason. True, but a gap and an unexplained gap are different.
* **Financial-sector caveat - 0 stocks, and a false claim on the public methodology page.**
  ``Financial_Sector_Caveat`` is written by ``factor_engine.add_financial_sector_caveat``, which
  only that module's own legacy end-to-end path calls. It is absent from
  ``05_final_scored.parquet``, so the payload published ``fin_caveat: false`` for all 498 - while
  ``SCREENER_OVERVIEW.md`` said "All financial-sector stocks receive a `Financial_Sector_Caveat`
  flag in the output". The field is deleted and the page corrected.

Every check here reads the **behaviour**: a browser render of the generator's own page, or the
generator's own methodology output - never the committed artifact, and never the presence of a
string in a source file (CLAUDE.md rule 10, and 0.11's "a claim whose check tests the plumbing
instead of the assertion is an unchecked claim").
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

try:
    from playwright import sync_api
except ImportError:  # pragma: no cover - the suite skips loudly without it
    sync_api = None


# --------------------------------------------------------------------------- the page, rendered
@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The generator's own page (not the committed file) with the published payloads beside it."""
    if sync_api is None:
        pytest.skip("MISSING DEP: playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip("MISSING DATA: no published %s" % f)
    d = tmp_path_factory.mktemp("deadbadge")
    (d / "index.html").write_text(
        gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v"),
        encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip("no usable chromium: %s" % exc)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function(
            "typeof renderProvenance === 'function' && typeof D !== 'undefined'", timeout=60000)
        yield pg
        b.close()
        assert not errors, errors


def _prov(page, stock: dict) -> str:
    """Render one stock through the function the drilldown calls, and read the panel's text."""
    s = {"metric_count": 25, "metric_total": 28, "data_source": "quarterly",
         "sector": "Industrials", "asof": {}, "inp": {}, "flags": {}}
    s.update(stock)
    # Render and read in one call, then read again (2026-10-29). Under full-suite load this module
    # once read EXPE's real panel - the stock the previous test rendered - instead of the synthetic
    # stock just rendered (NIGHTLY_LOG 2026-10-29 late). Two separate calls could not say whether the
    # render failed to land or something wrote the panel afterwards; these two reads do, and a
    # difference between them fails here, naming both, rather than as a wrong sentence below.
    now, later = page.evaluate("""s => { renderProvenance(s);
        const now = document.getElementById('modal-provenance').textContent;
        return new Promise(r => setTimeout(() => r([now, document.getElementById('modal-provenance').textContent]), 0)); }""", s)
    assert now == later, ("the provenance panel was rewritten after the render - "
                          f"rendered {now[:80]!r}, then held {later[:80]!r}")
    return re.sub(r"\s+", " ", now)


FOUR_QUARTERS = "the sum of the last four reported quarters"


# --------------------------------------------------------------- 1. the 4/3 annualisation
def test_the_annualised_stocks_get_their_own_sentence(page):
    """claims.py drilldown.ltm_annualized."""
    txt = _prov(page, {"flags": {"ltm_annualized": True,
                                 "ltm_annualized_labels": ["Operating Cash Flow", "Net Income"]}})
    assert "Three quarters scaled to twelve months" in txt
    # it names the lines, in the generator's own list style
    assert "Operating Cash Flow and Net Income are" in txt, txt
    # and says what the scaling assumes, which is the reason it is the weaker route
    assert "scaled by 4/3" in txt
    assert "assumes the quarter the source never published was an average one" in txt
    assert "weaker of the two routes" in txt


def test_the_annualised_sentence_works_without_labels(page):
    """The flag reached the payload for 103 runs without its labels; it must not print 'undefined'."""
    txt = _prov(page, {"flags": {"ltm_annualized": True}})
    assert "Three quarters scaled to twelve months" in txt
    assert "these figures are the three quarters summed" in txt
    assert "undefined" not in txt and "null" not in txt


def test_the_lead_does_not_claim_four_quarters_for_an_annualised_stock(page):
    """The defect itself: the live page asserted four summed quarters for these four stocks.

    Both directions are asserted, so narrowing the lead condition back fails here rather than
    passing quietly - and so does widening it to every stock, which would drop a true sentence.
    """
    ann = _prov(page, {"flags": {"ltm_annualized": True}})
    assert FOUR_QUARTERS not in ann, "the panel still claims four quarters were summed"
    assert "built for this company as described below" in ann

    plain = _prov(page, {"flags": {}})
    assert FOUR_QUARTERS in plain, "the general rule must still be stated for every other stock"
    assert "Three quarters scaled" not in plain

    # the other two constructions keep the behaviour the 2026-10-15 session shipped
    for flags in ({"ltm_rolled": True, "ltm_rolled_labels": ["Net Income"]},
                  {"q_rejected": "a duplicated quarter"}):
        assert FOUR_QUARTERS not in _prov(page, {"flags": flags})


def test_the_four_annualised_stocks_in_the_published_run_say_so(page):
    """The claim on the real payload, not a synthetic stock: every stock the engine annualised
    gets the sentence, and no other stock does."""
    detail = page.evaluate("Object.fromEntries(Object.entries(D.stock_detail)"
                           ".map(([t, s]) => [t, !!(s.flags || {}).ltm_annualized]))")
    ann = sorted(t for t, v in detail.items() if v)
    if not ann:
        pytest.skip("MISSING DATA: no annualised stock in the published run")
    for t in ann:
        txt = page.evaluate("t => { renderProvenance(D.stock_detail[t]); "
                            "return document.getElementById('modal-provenance').textContent; }", t)
        assert "Three quarters scaled to twelve months" in txt, t
        assert FOUR_QUARTERS not in txt, t
    other = next(t for t, v in detail.items() if not v)
    txt = page.evaluate("t => { renderProvenance(D.stock_detail[t]); "
                        "return document.getElementById('modal-provenance').textContent; }", other)
    assert "Three quarters scaled" not in txt, other


# --------------------------------------------------------------- 2. the substituted EV
def test_a_substituted_enterprise_value_is_stated_with_both_figures(page):
    """claims.py drilldown.ev_substituted."""
    txt = _prov(page, {"flags": {"ev_flag": True, "ev_api": 120e9},
                       "inp": {"ev_used": 80e9}})
    assert "Enterprise value recomputed" in txt
    # the three metrics it actually changes, so the reader knows the blast radius
    for m in ("EV/EBITDA", "EV/Sales", "free-cash-flow yield"):
        assert m in txt, m
    # both figures, so the 10% rule is checkable from the page
    assert "$120.0B" in txt and "$80.0B" in txt, txt
    assert "more than 10% away from market value plus debt less cash" in txt


def test_the_wider_tolerance_is_stated_for_a_financial(page):
    """Financials get 25%, because their debt includes deposits - stating 10% there would be false."""
    bank = _prov(page, {"sector": "Financials",
                        "flags": {"ev_flag": True, "is_bank": True, "ev_api": 120e9},
                        "inp": {"ev_used": 80e9}})
    assert "more than 25% away" in bank and "more than 10% away" not in bank
    assert "allowed 25% rather than the 10% used elsewhere" in bank
    assert "customer deposits" in bank
    # and the deposits clause must not appear on a stock the 25% does not apply to: printing it
    # on a utility's page explains a limit that is not its own (read on the page, 2026-10-27)
    util = _prov(page, {"sector": "Utilities",
                        "flags": {"ev_flag": True, "ev_api": 120e9}, "inp": {"ev_used": 80e9}})
    assert "more than 10% away" in util
    assert "customer deposits" not in util and "25%" not in util


def test_a_stock_whose_enterprise_value_stood_says_nothing(page):
    assert "Enterprise value recomputed" not in _prov(page, {"flags": {}})
    assert "Enterprise value recomputed" not in _prov(page, {"flags": {"ev_flag": False}})


def test_the_substituted_figure_survives_a_missing_counterpart(page):
    """18 stocks carry the flag; a run that lost `ev_api` must still state the rule, not 'undefined'."""
    txt = _prov(page, {"flags": {"ev_flag": True}, "inp": {}})
    assert "Enterprise value recomputed" in txt
    assert "undefined" not in txt and "NaN" not in txt and "Its figure was" not in txt


def test_the_engine_keeps_the_rejected_enterprise_value():
    """The sentence needs the figure the scorer threw away; `_ev_used` alone cannot show the rule
    fired. Asserted on the engine so the payload field cannot quietly stop being written."""
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    i = src.index('rec["_ev_flag"] = (')
    block = src[i:src.index("rec[\"_ev_used\"]", i)]
    assert 'rec["_ev_api"] = ev' in block, "the rejected EV is no longer kept"
    assert block.index('rec["_ev_api"] = ev') < block.index("ev = _ev_computed"), \
        "_ev_api must be captured before the substitution overwrites `ev`"


# --------------------------------------------------------------- 3. the unmeasurable beta
def test_an_unmeasurable_beta_says_why(page):
    """claims.py drilldown.beta_overlap."""
    txt = _prov(page, {"flags": {"beta_overlap_pct": 42.0}})
    assert "Beta could not be measured" in txt
    assert "Only 42% of this company" in txt
    assert "under the 80% the screener requires" in txt
    assert "Risk category" in txt


def test_the_overlap_sentence_fires_at_the_engine_s_own_bar(page):
    """The engine computes beta only at >= 80% overlap, so the sentence must appear strictly
    below 80 and not at it - the one point that separates `>` from `>=`
    (the 2026-10-25 mutation pass's lesson about a bar probed either side and never at it)."""
    assert "Beta could not be measured" in _prov(page, {"flags": {"beta_overlap_pct": 79.9}})
    assert "Beta could not be measured" not in _prov(page, {"flags": {"beta_overlap_pct": 80.0}})
    assert "Beta could not be measured" not in _prov(page, {"flags": {"beta_overlap_pct": 100.0}})
    # absent (no price history at all) is a different case and is not this sentence
    assert "Beta could not be measured" not in _prov(page, {"flags": {"beta_overlap_pct": None}})
    assert "Beta could not be measured" not in _prov(page, {"flags": {}})


def test_the_bar_in_the_sentence_is_the_bar_in_the_engine():
    """If the engine's 0.80 moves, the sentence's "80%" becomes false."""
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    assert "_overlap_ratio >= 0.80" in src, "the engine's beta overlap bar moved - update the sentence"
    gen = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert "fl.beta_overlap_pct < 80" in gen
    assert "under the 80% the screener requires" in gen


# --------------------------------------------------------------- 4. the caveat that never was
def test_the_overview_does_not_claim_the_caveat_flag():
    """claims.py overview.no_financial_sector_flag. Generated, so assert the generator's output,
    never the committed file (CLAUDE.md rule 10 - a hand-edit there is reverted by the 02:00 run)."""
    src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    old = ("All financial-sector stocks receive a `Financial_Sector_Caveat` flag in the output, "
           "reminding the user that financial companies require additional scrutiny")
    assert old not in src, "the false claim is back on the methodology page"
    i = src.index("There is no separate blanket flag for the sector")
    para = src[i:src.index("\n", i)]
    for phrase in ("0 of 498", "legacy end-to-end path",
                   "never called", "add_financial_sector_caveat"):
        assert phrase in para, phrase


def test_the_payload_no_longer_publishes_the_dead_field():
    """A field that is False for every stock on every run is worse than absent: a reader - or a
    later session - takes it for a measurement."""
    gen = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    # the exact code forms, not the word: the deletion is recorded in a comment that names it
    assert '"fin_caveat":' not in gen and "'fin_caveat':" not in gen
    assert 'row.get("Financial_Sector_Caveat")' not in gen
    # the field was emitted from exactly one place, so that one `flags` dict proves the payload
    i = gen.index('detail["flags"] = {')
    block = gen[i:gen.index("# --- Sector Peers ---", i)]
    live = [ln for ln in block.splitlines()
            if "fin_caveat" in ln and not ln.lstrip().startswith("#")]
    assert not live, "fin_caveat is published again: %r" % live
    caveat = [ln for ln in block.splitlines()
              if "Financial_Sector_Caveat" in ln and not ln.lstrip().startswith("#")]
    assert not caveat, "the dead column is read again: %r" % caveat
    # The committed payload is the 02:00 run's and predates this change, so it still carries the
    # field; the next full run clears it. The generator is what this can honestly assert.


def test_the_caveat_column_is_absent_from_the_scored_output():
    """The structural fact behind the correction, as a tripwire in both directions.

    If someone wires `add_financial_sector_caveat` into the pipeline, this fails and says so -
    because the methodology page's correction would then itself be out of date.
    """
    import run_context
    try:
        run_dir = run_context.latest_run_dir()
    except Exception:  # noqa: BLE001
        run_dir = None
    if run_dir is None or not (Path(run_dir) / "05_final_scored.parquet").exists():
        pytest.skip("MISSING DATA: no run with 05_final_scored.parquet")
    import pandas as pd
    df = pd.read_parquet(Path(run_dir) / "05_final_scored.parquet")
    assert "Financial_Sector_Caveat" not in df.columns, (
        "Financial_Sector_Caveat is in the scored output now - the flag is live, so the "
        "methodology page's paragraph and generate_dashboard's `flags` must be revisited")
    # and the thing that IS live and does change a score is still there
    assert "_is_bank_like" in df.columns
    assert int(df["_is_bank_like"].fillna(False).astype(bool).sum()) > 0


# --------------------------------------------------------------- the dead function itself
def test_the_dead_function_is_gone_and_records_where_its_badges_went():
    src = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    assert "function renderFlagsWarnings" not in src
    assert "getElementById('modal-flags')" not in src, "its container must not come back either"
    assert 'id="modal-flags"' not in src
    # the record of the audit stays with the code, so the next reader does not redo it
    i = src.index("`renderFlagsWarnings` stood here until 2026-10-27")
    note = src[i:i + 2000]
    for badge in ("value trap", "growth trap", "Beneish", "channel stuffing",
                  "stale financials", "EPS basis mismatch", "bank-like",
                  "LTM 3-of-4 annualised", "enterprise value substituted",
                  "beta overlap under 80%", "financial-sector caveat"):
        assert badge in note, badge


def test_the_seven_covered_badges_still_have_their_other_route():
    """The audit's "yes" column, encoded. Each of these is shown somewhere else; if that route
    is removed the information goes dark again, exactly as it did inside the dead function."""
    gen = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8")
    summ = (ROOT / "stock_summary.py").read_text(encoding="utf-8")
    # value trap, growth trap, Beneish, channel stuffing: the summary sentences and "the case against"
    for field in ("vt_severity", "gt_severity", "beneish_flag", "channel_stuffing"):
        assert field in summ, field
    assert "s.vt || s.gt || fl.beneish_flag || fl.channel_stuffing" in gen, \
        "the case-against block no longer collects the trap and manipulation flags"
    # stale financials: the summary, the case against, and the Filing Age badge
    assert "stale_data" in summ
    assert "missing || fl.stale_data || s.eps_mismatch" in gen
    assert "Filing Age: ' + age + ' days" in gen
    # EPS basis mismatch: the provenance badge and the data-quality KPI
    assert "EPS Mismatch (ratio: " in gen and "dqKpi('EPS Mismatch'" in gen
    # bank-like: the column label that changes with it
    assert "s.flags.is_bank ? 'ROE' : 'ROIC'" in gen


def test_the_published_run_still_matches_the_audit_s_counts():
    """The eleven counts in this module's docstring are the measurement the fixes were sized on.
    A later reader must be able to tell whether they still hold, so they are asserted loosely -
    present/absent, not exact - and the measurement script prints the exact ones."""
    p = ROOT / "dashboard_data.js"
    if not p.exists():
        pytest.skip("MISSING DATA: no published dashboard_data.js")
    txt = p.read_text(encoding="utf-8")
    d = json.loads(txt[txt.index("{"):txt.rstrip().rstrip(";").rindex("}") + 1])
    sd = d["stock_detail"]
    ann = [t for t, s in sd.items() if (s.get("flags") or {}).get("ltm_annualized")]
    ev = [t for t, s in sd.items() if (s.get("flags") or {}).get("ev_flag")]
    beta = [t for t, s in sd.items()
            if (s.get("flags") or {}).get("beta_overlap_pct") is not None
            and s["flags"]["beta_overlap_pct"] < 80]
    assert len(ann) + len(ev) + len(beta) > 0, (
        "no stock in the published run carries any of the three rehomed flags - if that is real "
        "the sentences are untested on live data and the counts in the docstring are stale")
