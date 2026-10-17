"""The market backdrop's sentences, against the records they describe.

Claims ``context.sahm``, ``context.macro_curve`` and ``context.macro_credit``.
Research: ``research/2026-10-17-macro-backdrop.md``.

**Why these tests recompute rather than compare.** Three sentences on the live panel were wrong
on 2026-10-17 in the same way: each attached a real citation to a *stronger or different*
statement than its source makes, and nothing checked the statement. The lesson already written
down for this layer (``plan/context-layer.md`` item 2) is that **a claim whose check tests the
plumbing instead of the assertion is an unchecked claim** - the old ``context.expected_move``
checks proved only that *a* date was carried through while the date itself was wrong for every
stock.

So these tests do not compare the sentence with a stored summary of the record. They rebuild the
record from the primary series - FRED's own CSVs, committed under ``tests/fixtures/fred/`` - and
assert the sentence agrees with what the arithmetic produces. If a future session widens a lag
range or drops a false alarm, the arithmetic says so.

The fixtures are a snapshot through September 2026 and are refreshed by
``python research/measurements/2026-10-17-fetch-macro-sources.py --fixtures`` after re-running
the two record scripts. A snapshot is the point: these are historical records, and a test that
re-downloaded them would fail offline and drift silently.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

import claims
import market_context

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fred"
THRESHOLD = 0.50


# --------------------------------------------------------------------------- helpers
def _series(sid: str) -> pd.Series:
    df = pd.read_csv(io.StringIO((FIXTURES / f"{sid}.csv").read_text(encoding="utf-8")))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def _episodes(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out, start, prev = [], None, None
    for d, v in flag.items():
        if v == 1 and start is None:
            start = d
        elif v == 0 and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, flag.index[-1]))
    return out


def _bond_equivalent(discount: pd.Series, days: int = 91) -> pd.Series:
    """The basis Estrella & Trubin (2006) use for the 3-month bill."""
    d = discount / 100.0
    return (365.0 * d / (360.0 - d * days)) * 100.0


@pytest.fixture(scope="module")
def jobs_text() -> str:
    out = market_context.readings({"UNRATE": {"last": 4.2}}, sahm=0.0, sahm_basis="real-time")
    return next(r["text"] for r in out if r["k"] == "jobs")


@pytest.fixture(scope="module")
def curve_text() -> str:
    out = market_context.readings({"T10Y3M": {"last": 0.82}}, sahm=None)
    return next(r["text"] for r in out if r["k"] == "curve")


@pytest.fixture(scope="module")
def credit_text() -> str:
    out = market_context.readings(
        {"BAA10Y": {"last": 1.44, "pct_10y": 2.0}}, sahm=None)
    return next(r["text"] for r in out if r["k"] == "credit")


# --------------------------------------------------------------------------- the register
def test_the_three_macro_claims_are_registered():
    ids = {c.id for c in claims.CLAIMS}
    assert {"context.sahm", "context.macro_curve", "context.macro_credit"} <= ids


def test_each_macro_claim_resolves_to_the_code_that_makes_it_true():
    for cid in ("context.sahm", "context.macro_curve", "context.macro_credit"):
        claim = next(c for c in claims.CLAIMS if c.id == cid)
        assert callable(claim.resolve()), cid


# --------------------------------------------------------------------------- the jobs sentence
def test_the_jobs_sentence_does_not_say_the_rule_marks_the_start(jobs_text):
    """The phrase that was false: the crossing came *after* the recession began in 6 of 8."""
    lowered = jobs_text.lower()
    assert "marked the start" not in lowered
    assert "around the start" in lowered
    assert "unemployment lags the cycle" in lowered


def test_the_jobs_sentence_names_both_false_alarms(jobs_text):
    assert "1976" in jobs_text and "2024" in jobs_text
    assert "without a recession following" in jobs_text


def test_the_jobs_sentence_says_what_sahm_proposed_the_rule_for(jobs_text):
    """Sahm (2019) proposed a trigger for automatic stimulus payments, not a recession forecast."""
    assert "not as a forecast" in jobs_text
    assert "stimulus" in jobs_text


def test_the_sahm_record_is_what_the_sentence_says_it_is():
    """Rebuild the record from FRED's own real-time series and the NBER indicator.

    Three assertions, one per clause of the sentence: every recession is caught, most crossings
    come after the recession has begun, and exactly two episodes have no recession following.
    """
    sahm, rec = _series("SAHMREALTIME"), _series("USREC")
    rec = rec.loc[rec.index >= sahm.index[0]]
    recessions = [r for r in _episodes(rec) if r[0].year >= 1970]
    assert len(recessions) == 8, "eight NBER recessions since 1970"

    lags = []
    for start, end in recessions:
        window = sahm.loc[(sahm.index >= start - pd.DateOffset(months=12)) & (sahm.index <= end)]
        over = window.loc[window >= THRESHOLD]
        assert len(over), f"no crossing around the {start:%Y-%m} recession"      # "every recession"
        first = over.index[0]
        peak = start - pd.DateOffset(months=1)
        lags.append((first.year - peak.year) * 12 + (first.month - peak.month))

    after = sum(1 for v in lags if v > 0)
    assert after == 6, f"crossings after the NBER peak: {after} of 8, lags {lags}"  # "usually in
    assert sum(lags) / len(lags) > 0, "the mean lag is positive - the rule lags the cycle"  # the
    #                                                                   first months rather than
    #                                                                   before it"

    off_cycle = []
    for start, end in _episodes((sahm >= THRESHOLD).astype(int)):
        ahead = rec.loc[(rec.index >= start - pd.DateOffset(months=1))
                        & (rec.index <= start + pd.DateOffset(months=12))]
        if not bool((ahead == 1).any()):
            off_cycle.append(start)
    years = sorted({d.year for d in off_cycle})
    assert years == [1976, 2024], f"episodes with no recession within 12 months: {years}"


# --------------------------------------------------------------------------- the curve sentence
def test_the_curve_sentence_gives_the_measured_lag_range(curve_text):
    """"roughly 6 to 24 months" was the defect: nothing on record leads by more than 17."""
    assert "6 to 17 months" in curve_text
    assert "24 months" not in curve_text


def test_the_curve_sentence_cites_the_paper_that_states_the_record(curve_text):
    """Estrella & Mishkin (1998) states neither the lag range nor a false-alarm count."""
    assert "Estrella & Trubin 2006" in curve_text
    assert "Mishkin" not in curve_text
    assert "no false signals" in curve_text
    assert "1966-67" in curve_text, "the one clear false alarm, and that it predates the period"


def test_the_curve_sentence_names_the_open_case(curve_text):
    assert "2022-24" in curve_text
    assert "open case" in curve_text


def test_the_curve_record_is_what_the_sentence_says_it_is():
    """All eight recessions since 1968, preceded by 6 to 17 months, on the source's bill basis."""
    gs10, tb3, rec = _series("GS10"), _series("TB3MS"), _series("USREC")
    spread = (gs10 - _bond_equivalent(tb3)).dropna()
    rec = rec.loc[rec.index >= spread.index[0]]
    recessions = [r for r in _episodes(rec) if r[0].year >= 1968]
    assert len(recessions) == 8, "eight NBER recessions since 1968"

    inversions = _episodes((spread < 0).astype(int))
    leads = []
    for start, _end in recessions:
        in12 = spread.loc[(spread.index < start) & (spread.index >= start - pd.DateOffset(months=12))]
        assert int((in12 < 0).sum()) >= 3, (
            f"{start:%Y-%m}: fewer than three negative monthly averages in the prior 12 "
            "- Estrella & Trubin (2006) p. 5 reports three or more for every recession")
        earliest = [e for e in inversions if start - pd.DateOffset(months=24) <= e[0] < start]
        assert earliest, f"no inversion in the 24 months before the {start:%Y-%m} recession"
        begin = earliest[0][0]
        leads.append((start.year - begin.year) * 12 + (start.month - begin.month))

    assert min(leads) == 6 and max(leads) == 17, f"measured leads {sorted(leads)}"

    # the 1966-67 false alarm, and that it sits before the period the sentence names
    pre = [e for e in inversions if e[0].year < 1968 and e[0].year >= 1966]
    assert pre, "the 1966-67 inversion is in the series"
    for start, end in pre:
        ahead = rec.loc[(rec.index >= start) & (rec.index <= end + pd.DateOffset(months=24))]
        assert not bool((ahead == 1).any()), f"{start:%Y-%m} was followed by a recession after all"

    # no completed episode since 1968 has failed; the 2022-24 one is open, not a failure
    failed = []
    for start, end in inversions:
        if start.year < 1968:
            continue
        closed = end + pd.DateOffset(months=24) <= rec.index[-1]
        ahead = rec.loc[(rec.index >= start) & (rec.index <= end + pd.DateOffset(months=24))]
        if closed and not bool((ahead == 1).any()):
            failed.append(f"{start:%Y-%m}")
    assert failed == [], f"episodes since 1968 with no recession in a closed window: {failed}"


# --------------------------------------------------------------------------- the credit sentence
def test_the_credit_sentence_names_the_premium_above_defaults(credit_text):
    """Gilchrist & Zakrajsek (2012): the predictive content is the premium, not expected defaults."""
    assert "premium lenders charge" in credit_text
    assert "above that" in credit_text
    assert "it is the second part that has predicted" in credit_text


def test_the_credit_sentence_carries_a_source(credit_text):
    """It had none at all before 2026-10-17, while making an empirical claim."""
    assert "Gilchrist & Zakrajsek (2012)" in credit_text


# --------------------------------------------------------------------------- the settled verdict
def test_no_macro_series_reaches_a_score():
    """Section 7 of the note: macro is one observation a month for the whole universe, so it is
    the wrong shape for a cross-sectional ranking. The module says so and nothing scores it."""
    doc = market_context.__doc__ or ""
    assert "No reading here is a candidate for the score" in doc
    source = (Path(market_context.__file__)).read_text(encoding="utf-8")
    assert "_ctx_" not in source, "the backdrop writes no per-stock context field"
