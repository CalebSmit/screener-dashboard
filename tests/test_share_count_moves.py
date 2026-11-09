"""The share count behind a market value, when it moves and the earnings have not caught up.

**The case.** Fox completed its acquisition of Roku on 2026-11-06. The source added the newly
issued Class A shares the next session, so FOXA's market value rose 34.3% with its price
unchanged: `earnings_yield` -25.5%, `fcf_yield` -23.1%, `ev_ebitda` and `ev_sales` +30.1%,
composite 63.19 -> 60.98, **rank 34 -> 60**. No price change, no new filing, and nothing in the
pipeline said so - a 499-name Spearman of 0.997 cannot see one stock move 26 places and
renormalisation preserves dispersion. `research/2026-11-09-share-counts-and-corporate-actions.md`
is the research; its section 4 is the open proposal this implements the reporting half of.

**The construct, and why it is not the one the note pre-registered.** The note proposed a
*run-to-run* delta in `marketCap / currentPrice` over 5%. The question it poses is a **state** -
the numerator predates the denominator for about three months - and a delta is an **event**.
Measured on KMB, whose count moved on 2026-11-04
(`research/measurements/2026-11-09-share-count-detector-review.py`): the delta named it on **1**
run; the comparison with its *filed* share count named it on **all 5** runs from 11-04 to 11-09.
The second comparison already existed (`sec_crosscheck.market_value_check`, in the health note
since 2026-10-14), so what was missing was never a detector - it was the names.

**And named by what is new, not by how far.** The over-10% set is mostly standing structure:
partnership units (IBKR, TKO, CVNA), splits after the filed quarter, two filers' own scale errors
(MCD, WAT). Measured on the 2026-11-09 run: a list of the **8 furthest** from their filed count
would not have named FOXA, which was **13th of 485**; against the published page the set is
exactly the news - **CHTR and FOXA new, 18 standing, CPT gone**. `test_a_nearer_new_gap_is_named_
over_a_further_standing_one` is that finding as a test, so the ordering cannot quietly go back to
magnitude.

Nothing here fails a run. It is a note and an `ATTENTION -` line, which is what the morning brief
surfaces - the `sec_crosscheck` precedent (CLAUDE.md 0.14): a measurement nobody has watched yet
is reported, and promoted to a gate only once its normal range is known.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402
import sec_crosscheck as X  # noqa: E402
import generate_dashboard as gd  # noqa: E402

FILED = 100.0            # every fixture company has filed a share count of 100
RUN_STAMP = "2026-11-09T02:00:00"


def _facts(tickers) -> pd.DataFrame:
    """One filed weighted-average share count each, for a quarter the run date can see."""
    return pd.DataFrame([{
        "ticker": t, "concept": "WeightedAverageNumberOfDilutedSharesOutstanding",
        "start": pd.Timestamp("2026-07-01"), "end": pd.Timestamp("2026-09-30"),
        "filed": pd.Timestamp("2026-11-01"), "form": "10-Q", "fp": "", "val": FILED,
    } for t in tickers])


def _payload(**ratios) -> dict:
    """``ratio`` per stock = implied share count / filed. Price is 1, so the market value is it."""
    return {"kpis": {"run_timestamp": RUN_STAMP},
            "stock_detail": {t: {"financials": {"market_cap": FILED * r}, "price": 1.0}
                             for t, r in ratios.items()}}


def _run(now: dict, was: dict | None):
    res = h.Result()
    tickers = set(now["stock_detail"]) | set((was or {}).get("stock_detail") or {})
    h.check_share_count_moves(now, res, was, facts=_facts(tickers))
    return res


def _attention(res):
    return [n for n in res.notes if n.startswith("ATTENTION - ")]


def _summary(res):
    return next(n for n in res.notes if n.startswith("share counts behind the market values:"))


# ------------------------------------------------------------------ what it reports
def test_a_new_gap_is_named_with_how_far_it_is_from_the_filed_count():
    res = _run(_payload(FOXA=1.311, AAA=1.0), _payload(FOXA=1.0, AAA=1.0))
    assert "1 of them new since the published page" in _summary(res)
    att = _attention(res)
    assert len(att) == 1 and "FOXA +31.1%" in att[0]
    assert "research/2026-11-09-share-counts-and-corporate-actions.md" in att[0]


def test_a_gap_the_published_page_already_shows_is_not_news():
    """IBKR's market value has implied ~3.8x its filed count every run for weeks - partnership
    units, recorded 2026-10-14. An alarm that fires on it every morning is an alarm nobody
    reads."""
    res = _run(_payload(IBKR=3.79, AAA=1.0), _payload(IBKR=3.78, AAA=1.0))
    assert "20 over" not in _summary(res)
    assert "1 over 10% from the filed count, 0 of them new" in _summary(res)
    assert _attention(res) == []


def test_a_gap_that_has_closed_is_reported_too():
    """A completed acquisition does not reverse, so a gap that closes is evidence the move was
    the source rather than the company - which is what CPT did between 2026-11-06 and 11-09, one
    of the three moves the research note could not establish."""
    res = _run(_payload(CPT=1.0, AAA=1.0), _payload(CPT=1.14, AAA=1.0))
    assert "1 no longer (CPT)" in _summary(res)
    assert _attention(res) == []


def test_a_nearer_new_gap_is_named_over_a_further_standing_one():
    """The measurement that chose novelty over magnitude, as a test.

    FOXA was **13th of 485** by distance from its filed count on 2026-11-09, behind MCD, IBKR,
    TKO, CVNA and the rest of the standing structure - so the obvious list would have missed the
    stock the whole research note is about.
    """
    now = _payload(MCD=900.0, IBKR=3.79, TKO=2.49, NEWS=1.12)
    was = _payload(MCD=900.0, IBKR=3.79, TKO=2.49, NEWS=1.00)
    res = _run(now, was)
    att = _attention(res)
    assert len(att) == 1
    assert "NEWS +12.0%" in att[0]
    for standing in ("MCD", "IBKR", "TKO"):
        assert standing not in att[0], f"{standing} is standing structure, not news"


def test_the_names_are_ordered_by_how_far_from_the_filed_count():
    res = _run(_payload(A=1.2, B=1.9, C=1.11), _payload(A=1.0, B=1.0, C=1.0))
    att = _attention(res)[0]
    assert att.index("B ") < att.index("A ") < att.index("C ")


def test_the_boundary_is_sec_crosschecks_own_ten_per_cent():
    """One line for one quantity. A second literal would be free to drift from the line the
    `inputs against SEC filings` note prints beside it (the 2026-10-31 switching-margin
    finding)."""
    assert X.FAR == 0.10
    assert _attention(_run(_payload(A=1.09), _payload(A=1.0))) == []
    assert len(_attention(_run(_payload(A=1.11), _payload(A=1.0)))) == 1
    src = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
    body = src.split("def check_share_count_moves")[1].split("\ndef ")[0]
    assert "X.FAR" in body and "0.10" not in body


def test_a_nominal_ten_per_cent_fires_and_that_is_the_existing_lines_behaviour():
    """Stated rather than hidden: the comparison is `abs(ratio - 1) > 0.10`, and a ratio of
    exactly 1.1 is marginally over 0.10 in binary floating point, so a nominal ten per cent
    fires. What matters is that it is the **same** comparison the `inputs against SEC filings`
    note prints beside it - the two counts are on the page together and must not disagree.
    """
    assert (1.1 - 1.0) > X.FAR                      # the float fact behind the behaviour
    assert len(_attention(_run(_payload(A=1.1), _payload(A=1.0)))) == 1
    # And the counts agree with sec_crosscheck's own, which is what the note beside it prints.
    now = _payload(A=1.1, B=1.09, C=3.0)
    mv = X.market_value_check(now, _facts(["A", "B", "C"]))
    theirs = int(((mv["ratio"] - 1).abs() > X.FAR).sum())
    res = _run(now, _payload(A=1.0, B=1.0, C=1.0))
    assert f"{theirs} over 10% from the filed count" in _summary(res)
    assert theirs == 2


# ------------------------------------------------------------------ the void condition
def test_past_the_pre_registered_void_condition_it_says_the_construct_is_wrong():
    """§5.5, written before the counts were in: over 3 a run date, sustained, this is measuring
    source churn. The response is to re-derive the detector, never to raise the line."""
    assert h.SHARE_COUNT_MOVE_VOID == 3
    many = {f"T{i}": 1.5 for i in range(4)}
    res = _run(_payload(**many), _payload(**{t: 1.0 for t in many}))
    void = [n for n in _attention(res) if "void condition" in n]
    assert len(void) == 1 and "re-derive the detector, do not raise the line" in void[0]
    assert res.healthy and not res.failures


def test_exactly_at_the_void_condition_it_does_not_fire():
    """`>` not `>=`, probed *at* the line - the one point that separates them, and the gap a
    test named `..._is_strict_and_configurable` hid on 2026-10-25 by probing either side."""
    three = {f"T{i}": 1.5 for i in range(h.SHARE_COUNT_MOVE_VOID)}
    res = _run(_payload(**three), _payload(**{t: 1.0 for t in three}))
    assert [n for n in _attention(res) if "void condition" in n] == []
    four = {f"T{i}": 1.5 for i in range(h.SHARE_COUNT_MOVE_VOID + 1)}
    res = _run(_payload(**four), _payload(**{t: 1.0 for t in four}))
    assert len([n for n in _attention(res) if "void condition" in n]) == 1


# ------------------------------------------------------------------ it never stops a publish
@pytest.mark.parametrize("baseline", [None])
def test_without_a_baseline_it_says_so_rather_than_guessing(baseline):
    res = h.Result()
    h.check_share_count_moves(_payload(A=1.5), res, baseline, facts=_facts(["A"]))
    assert res.healthy and not res.failures
    assert any("the published payload could not be read" in n for n in res.notes)
    assert _attention(res) == []


def test_nothing_comparable_is_a_note_not_a_verdict():
    """No filed share count for anything in the run: a gate that can only fail closed jams an
    unattended loop (the rule this file's siblings all keep)."""
    res = h.Result()
    h.check_share_count_moves(_payload(A=1.5), res, _payload(A=1.0),
                              facts=_facts([]).reindex(columns=_facts(["A"]).columns))
    assert res.healthy and not res.failures
    assert any(n.startswith("share-count moves not checked") for n in res.notes)


def test_a_broken_cache_is_reported_and_skipped():
    res = h.Result()
    h.check_share_count_moves(_payload(A=1.5), res, _payload(A=1.0), facts="not a frame")
    assert res.healthy and not res.failures
    assert any(n.startswith("share-count moves not checked - ") for n in res.notes)


def test_it_can_never_fail_a_run():
    src = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
    body = src.split("def check_share_count_moves")[1].split("\ndef ")[0]
    assert "res.fail(" not in body and "res.unevaluable" not in body


# ------------------------------------------------------------------ wiring
def test_main_runs_it_against_the_whole_published_payload():
    """`financials` has not been in the committed `dashboard_data.js` since 2026-11-08, so the
    baseline has to be the folded one or every standing gap reads as new on every run."""
    src = (ROOT / "scripts" / "check_run_health.py").read_text(encoding="utf-8")
    main = src.split("def main()")[1]
    assert "check_share_count_moves(payload, res, published_payload_whole())" in main
    assert main.index("check_inputs_against_filings(payload, res)") < \
        main.index("check_share_count_moves")


def test_the_whole_published_payload_is_the_committed_one_with_its_shards(monkeypatch):
    """It must read the shards from the **same ref**, not from the working tree - the working
    tree's payload is the run being judged (`published_payload`'s own reason)."""
    asked: list[str] = []

    def fake_git(ref):
        def read(name):
            asked.append(f"{ref}:{name}")
            if name == gd.DETAIL_FILE:
                return 'window.X = {"detail": {"AAA": {"financials": {"market_cap": 7.0}}}};'
            return None
        return read

    monkeypatch.setattr(h, "published_payload",
                        lambda ref="HEAD": {"stock_detail": {"AAA": {"price": 1.0}}})
    monkeypatch.setattr(h, "_git_reader", fake_git)
    whole = h.published_payload_whole("abc123")
    assert whole["stock_detail"]["AAA"]["financials"] == {"market_cap": 7.0}
    assert f"abc123:{gd.DETAIL_FILE}" in asked


def test_no_published_payload_means_no_baseline(monkeypatch):
    monkeypatch.setattr(h, "published_payload", lambda ref="HEAD": None)
    assert h.published_payload_whole() is None


def test_a_committed_shard_that_does_not_parse_means_no_baseline(monkeypatch):
    """A note, not a crash: the run is still publishable, it just cannot be called new."""
    monkeypatch.setattr(h, "published_payload",
                        lambda ref="HEAD": {"stock_detail": {"AAA": {"price": 1.0}}})
    monkeypatch.setattr(h, "_git_reader",
                        lambda ref: (lambda name: "window.X = {" if name == gd.DETAIL_FILE else None))
    assert h.published_payload_whole() is None
