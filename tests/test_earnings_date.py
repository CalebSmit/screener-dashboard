"""Next-earnings-date proximity: the one timing fact the evidence endorses.

`plan/dashboard-north-star.md` gap 4, open since 2026-08-10 and verified
available 2026-09-16. The dashboard could score a company on 44 metrics without
telling a reader when it next reports.

**Why this fact and not another.** In Akepanidtaworn, Di Mascio, Imas & Schmidt
(2023, *Journal of Finance* 78(6)), sells executed on a holding's
earnings-announcement day beat non-announcement-day sells by more than
**+150 bp/year** and are the *only* sells in 4.4 million trades that beat a
random-disposal counterfactual. The paper attributes that to attention, not
skill: an announcement is an exogenous, pre-scheduled reason to re-examine a
position. The same paper is already why the holdings panel refuses to rank by
size of move, so this is the positive half of a source the tool already relies
on.

It also matters mechanically for *this* screener. The fundamental inputs come
from filings and barely move between them - measured across a month of this
repo's own runs, the largest one-month Quality-score move was **one stock in
500**, against 34% for Risk. The report date is when Valuation, Quality and
Growth are actually replaced.

The behaviours pinned here, in rough order of how much damage getting them
wrong would do:

1. **It is never scored.** A proximity-to-earnings number reaching `raw`/`pct`
   is a new factor smuggled in as a UI feature. Same guard `about` carries.
2. **An estimated date is labelled every time.** Measured over an 80-name
   sample on 2026-09-29, **34 (42.5%)** of next dates were provider estimates.
   A missing flag reading as "confirmed" would overstate four dates in ten.
3. **A date at or before the run date is dropped, never relabelled.** It cannot
   honestly be called "next" and there is no evidence for what else it is.
4. **`earningsTimestamp` is not read at all.** Measured live 2026-09-29, it
   holds the *last* report for AAPL (2026-07-30) and EXPE (2026-08-05) and the
   *next* for HST, JPM and NVDA. No label is true of every row.
5. **The horizon is anchored to the run, not the reader's clock.** The summary
   is baked into the payload, so a bare "in 36 days" would decay into a
   falsehood over a weekend while reading as current.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_dashboard as G  # noqa: E402
import stock_summary as S  # noqa: E402

RUN_DATE = "2026-09-29"


def _epoch(iso: str) -> float:
    """A UTC-midnight UNIX timestamp for an ISO date, as yfinance returns."""
    d = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
    return d.timestamp()


def _detail(day: str | None = "2026-11-04", est: bool = False,
            end: str | None = None) -> dict:
    earn: dict = {}
    if day:
        earn["d"] = day
        earn["est"] = est
        if end:
            earn["end"] = end
    return {"earn": earn} if earn else {}


# ---------------------------------------------------------------------------
# 1. The sentence
# ---------------------------------------------------------------------------

def test_states_the_date_and_the_horizon():
    s = S._sentence_earnings(_detail("2026-11-04"), RUN_DATE)
    assert "4 Nov 2026" in s
    assert "36 days after this run" in s


def test_horizon_is_anchored_to_the_run_not_the_reader():
    """Property 5. The phrasing must name its own anchor.

    A bare "in 36 days" is baked into a payload that is republished on
    weekdays only, so across a weekend it silently becomes wrong while still
    reading as current. Naming the run makes a stale page obviously stale.
    """
    s = S._sentence_earnings(_detail("2026-11-04"), RUN_DATE)
    assert "after this run" in s
    assert "in 36 days" not in s


@pytest.mark.parametrize("day,expected", [
    ("2026-09-29", "the day of this run"),
    ("2026-09-30", "the day after this run"),
    ("2026-10-01", "2 days after this run"),
])
def test_near_horizons_read_naturally(day, expected):
    assert expected in S._sentence_earnings(_detail(day), RUN_DATE)


def test_estimated_date_is_labelled():
    """Property 2 - the single most load-bearing behaviour here."""
    s = S._sentence_earnings(_detail("2026-11-05", est=True), RUN_DATE)
    assert "provider estimate" in s
    assert "not a confirmed date" in s or "rather than a confirmed date" in s


def test_confirmed_date_carries_no_estimate_caveat():
    s = S._sentence_earnings(_detail("2026-11-05", est=False), RUN_DATE)
    assert "estimate" not in s.lower()


def test_estimated_and_confirmed_differ_only_by_the_caveat():
    """The caveat must be additive. If the two sentences diverged elsewhere,
    a reader could not tell which part of the difference was the uncertainty."""
    conf = S._sentence_earnings(_detail("2026-11-05", est=False), RUN_DATE)
    est = S._sentence_earnings(_detail("2026-11-05", est=True), RUN_DATE)
    assert est.startswith(conf[:-1])


def test_window_is_stated_as_a_range():
    s = S._sentence_earnings(
        _detail("2026-11-04", end="2026-11-06"), RUN_DATE)
    assert "between 4 Nov 2026 and 6 Nov 2026" in s
    # The horizon still counts from the start of the window, not its end.
    assert "36 days after this run" in s


def test_equal_end_does_not_render_a_degenerate_range():
    s = S._sentence_earnings(
        _detail("2026-11-04", end="2026-11-04"), RUN_DATE)
    assert "between" not in s
    assert "on 4 Nov 2026" in s


@pytest.mark.parametrize("day", ["2026-09-28", "2026-08-01", "2025-01-01"])
def test_past_dates_are_dropped_never_relabelled(day):
    """Property 3. A stale date must produce silence, not a wrong claim."""
    assert S._sentence_earnings(_detail(day), RUN_DATE) is None


def test_missing_date_is_silent():
    assert S._sentence_earnings(_detail(None), RUN_DATE) is None
    assert S._sentence_earnings({}, RUN_DATE) is None


def test_missing_run_date_is_silent():
    """Without an anchor the horizon cannot be stated, and the horizon is the
    decision-relevant half. A bare date with no sense of distance is not worth
    inventing a fallback clock for."""
    assert S._sentence_earnings(_detail("2026-11-04"), None) is None


@pytest.mark.parametrize("bad", ["garbage", "", "2026-13-45", "11/04/2026"])
def test_unparseable_dates_are_silent_not_fatal(bad):
    assert S._sentence_earnings(_detail(bad), RUN_DATE) is None
    assert S._sentence_earnings(_detail("2026-11-04"), bad) is None


def test_unparseable_window_end_falls_back_to_the_start():
    s = S._sentence_earnings(_detail("2026-11-04", end="nonsense"), RUN_DATE)
    assert "on 4 Nov 2026" in s


def test_end_before_start_is_ignored():
    """A provider inversion must not render "between 4 Nov and 1 Nov"."""
    s = S._sentence_earnings(_detail("2026-11-04", end="2026-11-01"), RUN_DATE)
    assert "between" not in s
    assert "on 4 Nov 2026" in s


def test_leading_zero_is_not_printed():
    """strftime('%-d') is not portable to Windows, which is where this runs."""
    s = S._sentence_earnings(_detail("2026-11-04"), RUN_DATE)
    assert "04 Nov" not in s
    assert "4 Nov" in s


# ---------------------------------------------------------------------------
# 2. It never becomes advice
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("day,est", [
    ("2026-09-29", False), ("2026-09-30", True), ("2026-10-15", False),
    ("2027-01-02", True),
])
def test_no_advice_language_at_any_horizon(day, est):
    """The tempting version of this feature is a countdown that shouts as the
    date nears. `BANNED_TERMS` is the machine-checkable form of the north-star
    line; the sentence must clear it whether the report is today or in a year."""
    s = S._sentence_earnings(_detail(day, est=est), RUN_DATE)
    assert S.advice_terms_in(s) == []


def test_wording_does_not_change_with_proximity():
    """A near date and a far one must differ only in the number of days.

    The evidence says announcement days are when attention is well spent; it
    does not say a near report is good or bad news. Escalating the wording as
    the date approaches would assert the second.
    """
    near = S._sentence_earnings(_detail("2026-10-01"), RUN_DATE)
    far = S._sentence_earnings(_detail("2027-01-01"), RUN_DATE)
    assert near.replace("2 days", "X").replace("1 Oct 2026", "D") == \
        far.replace("94 days", "X").replace("1 Jan 2027", "D")


# ---------------------------------------------------------------------------
# 3. Wiring into the summary
# ---------------------------------------------------------------------------

def _full_detail(**kw) -> dict:
    d = {
        "rank": 3, "composite": 71.0, "sector": "Energy",
        "cat_scores": {c: 60.0 for c in G.CATEGORIES},
        "contrib": {c: 7.0 for c in G.CATEGORIES},
        "raw": {}, "pct": {},
        "metric_count": 18, "metric_total": 18,
    }
    d.update(kw)
    return d


def test_summary_carries_the_earnings_kind():
    out = S.build_summary(
        _full_detail(earn={"d": "2026-11-04", "est": False}),
        universe_size=502, metric_meta={}, metric_weights={},
        run_date=RUN_DATE)
    kinds = [f["k"] for f in out]
    assert "earnings" in kinds


def test_summary_omits_the_kind_when_there_is_no_date():
    out = S.build_summary(
        _full_detail(), universe_size=502, metric_meta={}, metric_weights={},
        run_date=RUN_DATE)
    assert "earnings" not in [f["k"] for f in out]


def test_run_date_defaults_to_none_so_old_callers_do_not_break():
    """`run_date` is keyword-only with a default: a caller that has not been
    updated gets a summary without the sentence, not a TypeError."""
    out = S.build_summary(
        _full_detail(earn={"d": "2026-11-04", "est": False}),
        universe_size=502, metric_meta={}, metric_weights={})
    assert "earnings" not in [f["k"] for f in out]


def test_earnings_follows_confidence():
    """Both say how far the score can be trusted to stand - `confidence` what it
    rests on, `earnings` when those inputs are next replaced. Splitting them
    would put a coverage caveat between a fact and its qualifier."""
    out = S.build_summary(
        _full_detail(earn={"d": "2026-11-04", "est": False},
                     metric_count=12, metric_total=18),
        universe_size=502, metric_meta={}, metric_weights={},
        run_date=RUN_DATE)
    kinds = [f["k"] for f in out]
    assert kinds.index("earnings") == kinds.index("confidence") + 1


def test_summary_survives_a_malformed_earn_block():
    """A provider oddity must cost the sentence, not the drilldown."""
    for bad in [{"d": None}, {"est": True}, {"d": 12345}, {}]:
        out = S.build_summary(
            _full_detail(earn=bad), universe_size=502, metric_meta={},
            metric_weights={}, run_date=RUN_DATE)
        assert [f["k"] for f in out]  # still a usable summary
        assert "earnings" not in [f["k"] for f in out]


# ---------------------------------------------------------------------------
# 4. The payload block
# ---------------------------------------------------------------------------

def test_epoch_converts_to_an_iso_date():
    assert G._epoch_to_date(_epoch("2026-11-04")) == "2026-11-04"


@pytest.mark.parametrize("bad", [None, float("nan"), "", "abc", [], {}])
def test_epoch_rejects_junk_without_raising(bad):
    assert G._epoch_to_date(bad) is None


def test_block_omits_end_when_it_equals_the_start():
    """Sampled over 80 names, start and end were always equal. Carrying a
    duplicate date on every stock would be ~500 wasted payload entries."""
    row = {"_earn_start": _epoch("2026-11-04"),
           "_earn_end": _epoch("2026-11-04"), "_earn_est": False}
    block = G._earnings_block(row)
    assert block == {"d": "2026-11-04", "est": False}
    assert "end" not in block


def test_block_keeps_a_real_window():
    row = {"_earn_start": _epoch("2026-11-04"),
           "_earn_end": _epoch("2026-11-06"), "_earn_est": True}
    assert G._earnings_block(row) == {
        "d": "2026-11-04", "end": "2026-11-06", "est": True}


def test_block_is_none_without_a_start():
    assert G._earnings_block({"_earn_end": _epoch("2026-11-06")}) is None
    assert G._earnings_block({}) is None
    assert G._earnings_block({"_earn_start": None}) is None


def test_est_is_always_present():
    """Property 2 at the payload layer. A block without `est` would render as
    confirmed in the front end, which for 42.5% of names would be false."""
    row = {"_earn_start": _epoch("2026-11-04")}
    block = G._earnings_block(row)
    assert "est" in block and block["est"] is False


def test_missing_est_flag_does_not_become_true():
    row = {"_earn_start": _epoch("2026-11-04"), "_earn_est": float("nan")}
    assert G._earnings_block(row)["est"] is False


def test_est_is_a_real_bool_not_a_numpy_scalar():
    """`json.dumps` cannot serialise `numpy.bool_`, and the payload is written
    with the stdlib encoder. A pandas-sourced flag must be coerced."""
    import numpy as np
    row = {"_earn_start": _epoch("2026-11-04"), "_earn_est": np.bool_(True)}
    block = G._earnings_block(row)
    assert type(block["est"]) is bool
    import json
    json.dumps(block)


# ---------------------------------------------------------------------------
# 5. Display-only - it must never be scored
# ---------------------------------------------------------------------------

EARN_FIELDS = ["earningsTimestampStart", "earningsTimestampEnd",
               "isEarningsDateEstimate", "_earn_start", "_earn_end",
               "_earn_est", "earn"]


def test_earnings_fields_are_not_scored_metrics():
    """Property 1. `METRIC_COLS` is the registry the scorer iterates; anything
    absent from it cannot reach a category score."""
    from factor_engine import METRIC_COLS
    for f in EARN_FIELDS:
        assert f not in METRIC_COLS
        assert f"{f}_pct" not in METRIC_COLS


def test_earnings_fields_carry_no_weight():
    import yaml
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    weighted = set()
    for cat in (cfg.get("metric_weights") or {}).values():
        weighted.update(cat or {})
    for cat in (cfg.get("bank_metric_weights") or {}).values():
        weighted.update(cat or {})
    for f in EARN_FIELDS:
        assert f not in weighted


def test_earnings_fields_have_no_scoring_direction():
    """`METRIC_DIR` is what would let a percentile be computed for a field."""
    from factor_engine import METRIC_DIR
    for f in EARN_FIELDS:
        assert f not in METRIC_DIR


def test_the_fetcher_does_not_read_the_ambiguous_timestamp():
    """Property 4. `earningsTimestamp` means different things for different
    tickers (AAPL/EXPE last, HST/JPM/NVDA next), so it must stay uncaptured -
    a future session reaching for the obvious-looking field is the risk."""
    src = (Path(__file__).resolve().parent.parent / "factor_engine.py").read_text()
    assert 'rec["earningsTimestamp"]' not in src
    assert '_safe(info, "earningsTimestamp")' not in src
    # ...while the two that are unambiguous are read.
    assert 'rec["earningsTimestampStart"]' in src
    assert 'rec["earningsTimestampEnd"]' in src
    assert 'rec["isEarningsDateEstimate"]' in src


def test_live_payload_keeps_earnings_out_of_raw_and_pct():
    """The same assertion `about`/`industry` carry, run against whatever the
    last build actually published. Skips cleanly before the first build that
    carries the field."""
    js = Path(__file__).resolve().parent.parent / "dashboard_data.js"
    if not js.exists():
        pytest.skip("no published payload")
    import json as _json
    text = js.read_text(encoding="utf-8")
    start = text.find("{")
    data = _json.loads(text[start:text.rindex("}") + 1])
    detail = data.get("stock_detail") or {}
    seen = 0
    for t, d in detail.items():
        if "earn" not in d:
            continue
        seen += 1
        assert set(d["earn"]) <= {"d", "end", "est"}, t
        for f in EARN_FIELDS:
            assert f not in (d.get("raw") or {}), t
            assert f not in (d.get("pct") or {}), t
    if seen:
        # Whatever is published must be a date, in the future as of that run.
        for t, d in detail.items():
            if "earn" in d:
                datetime.fromisoformat(d["earn"]["d"])


# ---------------------------------------------------------------------------
# 6. The front end
# ---------------------------------------------------------------------------

def test_holdings_rows_render_the_earnings_note():
    """This is the surface the +150 bp/year result actually points at: a
    scheduled, external reason to re-examine something you own."""
    src = (Path(__file__).resolve().parent.parent
           / "generate_dashboard.py").read_text()
    idx = src.find("const HOLDINGS_FACTS")
    assert idx != -1
    line = src[idx:src.find("\n", idx)]
    assert "'earnings'" in line


def test_the_footnote_sources_the_claim():
    """Every arguable claim on the holdings panel is sourced in the footnote,
    because the constraints look arbitrary without their evidence."""
    src = (Path(__file__).resolve().parent.parent
           / "generate_dashboard.py").read_text()
    assert "Why the next earnings date is here" in src
    assert "150 basis points a year" in src
    assert "42.5%" in src
    assert "never scored" in src


def test_holdings_row_actually_renders_the_note(tmp_path):
    """Membership in `HOLDINGS_FACTS` is necessary but not sufficient - the
    renderer also strips the kind through a character class before using it as
    a CSS suffix. Drive the emitted script and read the output.
    """
    hp = pytest.importorskip("tests.test_holdings_panel",
                             reason="node harness unavailable")
    if hp.NODE is None:
        pytest.skip("node not available")
    html = (Path(__file__).resolve().parent.parent / "index.html").read_text(
        encoding="utf-8")
    import re as _re
    blocks = _re.findall(r"<script>(.*?)</script>", html, _re.S)
    assert len(blocks) == 1
    out = hp._run_js(blocks[0], """
        A.D.stock_detail['AAA'].summary.push(
          {k: 'earnings', t: 'It is scheduled to report earnings on 4 Nov 2026, 36 days after this run.'});
        A.initHoldings();
        A.addHolding('AAA');
        const html = globalThis.__els['holdings-body'].innerHTML;
        console.log(JSON.stringify({
            note: html.includes('holding-note-earnings'),
            text: html.includes('scheduled to report earnings on 4 Nov 2026'),
            peers: html.includes('must not appear'),
        }));
    """, tmp_path)
    assert out["note"] is True
    assert out["text"] is True
    # The unrelated constraint this could plausibly break: non-review facts
    # stay off the holdings row.
    assert out["peers"] is False


def test_the_note_has_no_urgency_styling():
    """No red, no badge, no ramp. See `test_wording_does_not_change_with_proximity`
    - the same argument at the CSS layer."""
    src = (Path(__file__).resolve().parent.parent
           / "generate_dashboard.py").read_text()
    idx = src.find(".holding-note-earnings")
    assert idx != -1
    rule = src[idx:src.find("}", idx)]
    for banned in ["--red", "--amber", "bold", "700", "uppercase"]:
        assert banned not in rule
