"""Input-availability churn: telling a measurement change from a company change.

`research/2026-09-14-sell-discipline-and-hold-bands.md` §8.3. When a metric
percentile flips between present and absent from one run to the next, its
category renormalises over a different metric set and the score moves as a
matter of arithmetic - no company event required. `CLAUDE.md` priority 1.5
records the worked case: FCX's growth score went 68.3 -> 42.5 -> 68.3 across
three runs because `forward_eps_growth` and `peg_ratio` went NaN and came back.
That was investigated as a suspected defect and turned out to be the pipeline
working exactly as designed - which is precisely why the *surface* has to say
so, because nothing downstream could tell the two apart.

Measured over 12,044 ticker-transitions across 24 run-pairs, the median
|rank change| is 6 with no churn, 7 with one metric changed, and **21** with two
or three. So the mechanism is arithmetic (needs no significance test) and only
its size was ever in question (it was measured).

The behaviours pinned here, in rough order of how much damage getting them wrong
would do:

1. **"Cannot tell" never renders as "nothing changed."** A snapshot predating
   the percentile schema, or a ticker absent from one run, yields None - not 0.
2. **Only columns both runs carry are compared.** The schema has grown over
   time; counting a column that did not exist yet as a metric that went missing
   would flag the entire universe on the day a metric was added.
3. **The caveat arms at two metrics, not one.** One is indistinguishable from
   noise and firing on it would mark 4.93% of transitions to say nothing.
4. **It is worded as a caveat on the comparison, never as a reason to act.**
   Churn scatters ranks near-symmetrically (52.4% worse off against a 44.6%
   base rate); presenting it as deterioration would manufacture the exact sell
   trigger Akepanidtaworn et al. (2023) find destroys value.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import history as H  # noqa: E402
import stock_summary as S  # noqa: E402


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

METRICS = ["ev_ebitda_pct", "roic_pct", "peg_ratio_pct",
           "forward_eps_growth_pct", "return_6m_pct"]


def _df(ranks: dict[str, int], missing: dict[str, list[str]] | None = None,
        metric_cols: list[str] | None = None) -> pd.DataFrame:
    """A snapshot frame, with named metric percentiles set NaN per ticker.

    ``metric_cols=[]`` reproduces the pre-2026-03-09 schema: 15 columns, no
    percentiles at all.
    """
    missing = missing or {}
    cols = METRICS if metric_cols is None else metric_cols
    n = len(ranks)
    rows = []
    for ticker, rank in ranks.items():
        row = {"Ticker": ticker, "Sector": "Tech", "Rank": rank,
               "Composite": round(100.0 * (n - rank + 1) / n, 2)}
        for cat in H.CATEGORIES:
            row[f"{cat}_score"] = 50.0
        for col in cols:
            row[col] = None if col in missing.get(ticker, []) else 50.0
        rows.append(row)
    return pd.DataFrame(rows)


def _snap(date: str, ranks, missing=None, metric_cols=None) -> H.RunSnapshot:
    return H._to_run_snapshot(date, _df(ranks, missing, metric_cols))


RANKS = {"AAA": 1, "BBB": 2, "CCC": 3}


def _delta(lost: int, gained: int, key: str = "m1", dr: int = -40) -> dict:
    # `cat` is present so `_sentence_change_driver` also fires - the ordering
    # tests need both sentences to exist to assert this one follows them.
    return {key: {"dr": dr, "dc": -4.2, "cat": {"growth": -9.0, "risk": 2.0},
                  "ch": [lost, gained]}}


COMPARE = {"m1": {"date": "2026-08-20", "gap_days": 28},
           "prev": {"date": "2026-09-16", "gap_days": 1}}


# ---------------------------------------------------------------------------
# metric_pct_columns - which columns count as metrics
# ---------------------------------------------------------------------------

def test_metric_columns_found():
    assert H.metric_pct_columns(_df(RANKS)) == sorted(METRICS)


def test_metric_columns_empty_for_pre_schema_snapshot():
    """Snapshots before 2026-03-09 carry 15 columns and no percentiles."""
    assert H.metric_pct_columns(_df(RANKS, metric_cols=[])) == []


# Pinned as literals, not read from `history`, so emptying the module's own
# exclusion set fails these rather than silently reparametrising them away.
NON_METRICS = ["_beta_overlap_pct", "portfolio_turnover_pct"]


@pytest.mark.parametrize("col", NON_METRICS)
def test_non_metric_pct_columns_excluded(col):
    """Both end in `_pct` and neither is a metric percentile. `_beta_overlap_pct`
    is a diagnostic and `portfolio_turnover_pct` is one number for the whole
    run, so counting either would put churn on every ticker at once."""
    df = _df(RANKS)
    df[col] = 1.0
    assert col not in H.metric_pct_columns(df)


def test_exclusion_set_is_exactly_those_two():
    assert H.NON_METRIC_PCT_COLS == frozenset(NON_METRICS)


def test_exclusion_list_matches_measurement_script():
    """The script that produced the numbers in the research note hard-codes the
    same two columns. A divergence would mean the shipped flag and its published
    justification measure different things, and would be silent."""
    script = (Path(__file__).resolve().parent.parent / "research" / "measurements"
              / "2026-09-16-hold-band-and-input-churn.py")
    text = script.read_text(encoding="utf-8")
    for col in NON_METRICS:
        assert col in text, f"{col} excluded in history.py but not in the script"


def test_metric_columns_sorted():
    """Sorted order makes the zip in `_metric_availability` reproducible."""
    df = _df(RANKS)
    assert H.metric_pct_columns(df) == sorted(H.metric_pct_columns(df))


# ---------------------------------------------------------------------------
# RunSnapshot carries availability
# ---------------------------------------------------------------------------

def test_snapshot_records_missing_metrics():
    snap = _snap("2026-09-01", RANKS, {"BBB": ["roic_pct", "peg_ratio_pct"]})
    assert snap.missing_metrics["BBB"] == frozenset({"roic_pct", "peg_ratio_pct"})


def test_snapshot_omits_tickers_with_nothing_missing():
    """Storing the missing set rather than the available one is what keeps this
    from becoming the largest thing the history holds."""
    snap = _snap("2026-09-01", RANKS, {"BBB": ["roic_pct"]})
    assert set(snap.missing_metrics) == {"BBB"}


def test_snapshot_metric_cols_recorded():
    snap = _snap("2026-09-01", RANKS)
    assert snap.metric_cols == frozenset(METRICS)


def test_snapshot_pre_schema_has_no_metric_cols():
    snap = _snap("2026-02-20", RANKS, metric_cols=[])
    assert snap.metric_cols == frozenset()
    assert snap.missing_metrics == {}


def test_runsnapshot_defaults_keep_hand_built_instances_valid():
    """The other test modules build RunSnapshot directly. Both new fields
    default, so churn degrades to unavailable rather than raising."""
    snap = H.RunSnapshot(date="2026-09-01", ranks={"AAA": 1},
                         composites={"AAA": 90.0}, cat_scores={"AAA": {}})
    assert snap.metric_cols == frozenset()
    assert snap.missing_metrics == {}


# ---------------------------------------------------------------------------
# input_churn - the arithmetic
# ---------------------------------------------------------------------------

def test_churn_counts_a_lost_metric():
    base = _snap("2026-08-20", RANKS)
    cur = _snap("2026-09-17", RANKS, {"AAA": ["roic_pct"]})
    assert H.input_churn(cur, base, "AAA") == (1, 0)


def test_churn_counts_a_gained_metric():
    base = _snap("2026-08-20", RANKS, {"AAA": ["roic_pct"]})
    cur = _snap("2026-09-17", RANKS)
    assert H.input_churn(cur, base, "AAA") == (0, 1)


def test_churn_counts_both_directions_separately():
    """The net count can be zero while two metrics changed. Counting net
    difference instead of lost+gained would miss exactly the case where one
    metric drops out as another returns."""
    base = _snap("2026-08-20", RANKS, {"AAA": ["roic_pct"]})
    cur = _snap("2026-09-17", RANKS, {"AAA": ["peg_ratio_pct"]})
    assert H.input_churn(cur, base, "AAA") == (1, 1)


def test_stable_inputs_report_zero_not_none():
    """Zero and None mean different things and both are load-bearing."""
    base = _snap("2026-08-20", RANKS, {"AAA": ["roic_pct"]})
    cur = _snap("2026-09-17", RANKS, {"AAA": ["roic_pct"]})
    assert H.input_churn(cur, base, "AAA") == (0, 0)


def test_churn_is_per_ticker():
    base = _snap("2026-08-20", RANKS)
    cur = _snap("2026-09-17", RANKS, {"BBB": ["roic_pct", "peg_ratio_pct"]})
    assert H.input_churn(cur, base, "AAA") == (0, 0)
    assert H.input_churn(cur, base, "BBB") == (2, 0)


# ---------------------------------------------------------------------------
# input_churn - "cannot tell" must not read as "nothing changed"
# ---------------------------------------------------------------------------

def test_churn_none_when_baseline_predates_the_schema():
    base = _snap("2026-02-20", RANKS, metric_cols=[])
    cur = _snap("2026-09-17", RANKS, {"AAA": ["roic_pct"]})
    assert H.input_churn(cur, base, "AAA") is None


def test_churn_none_when_current_has_no_metric_columns():
    base = _snap("2026-08-20", RANKS)
    cur = _snap("2026-09-17", RANKS, metric_cols=[])
    assert H.input_churn(cur, base, "AAA") is None


def test_churn_none_for_a_ticker_absent_from_the_baseline():
    base = _snap("2026-08-20", {"AAA": 1, "BBB": 2})
    cur = _snap("2026-09-17", RANKS)
    assert H.input_churn(cur, base, "CCC") is None


def test_churn_none_for_a_ticker_absent_from_the_current_run():
    base = _snap("2026-08-20", RANKS)
    cur = _snap("2026-09-17", {"AAA": 1, "BBB": 2})
    assert H.input_churn(cur, base, "CCC") is None


# ---------------------------------------------------------------------------
# input_churn - schema growth must not masquerade as churn
# ---------------------------------------------------------------------------

def test_a_newly_added_metric_column_is_not_churn():
    """`fy1_revision_3m_pct` appears part-way through the real snapshot
    directory. If a column present in only one run counted, every ticker in the
    universe would flag on the day a metric was added - a caveat on all 500
    names, which is the same as no caveat at all."""
    base = _snap("2026-08-20", RANKS, metric_cols=METRICS[:3])
    cur = _snap("2026-09-17", RANKS, metric_cols=METRICS)
    assert H.input_churn(cur, base, "AAA") == (0, 0)


def test_a_removed_metric_column_is_not_churn():
    base = _snap("2026-08-20", RANKS, metric_cols=METRICS)
    cur = _snap("2026-09-17", RANKS, metric_cols=METRICS[:3])
    assert H.input_churn(cur, base, "AAA") == (0, 0)


def test_churn_within_shared_columns_survives_schema_growth():
    """The intersection rule must not also suppress real churn."""
    base = _snap("2026-08-20", RANKS, {"AAA": ["roic_pct"]},
                 metric_cols=METRICS[:3])
    cur = _snap("2026-09-17", RANKS, {"AAA": ["ev_ebitda_pct", "peg_ratio_pct"]},
                metric_cols=METRICS)
    # roic returned; ev_ebitda and peg went missing. All three are shared.
    assert H.input_churn(cur, base, "AAA") == (2, 1)


# ---------------------------------------------------------------------------
# _compare / build_history wiring
# ---------------------------------------------------------------------------

def test_compare_emits_ch_when_inputs_changed():
    base = _snap("2026-08-20", RANKS)
    cur = _snap("2026-09-17", RANKS, {"AAA": ["roic_pct", "peg_ratio_pct"]})
    assert H._compare(cur, base)["AAA"]["ch"] == [2, 0]


def test_compare_omits_ch_when_inputs_are_stable():
    """~95% of tickers are stable, so the key must not be paid for on them."""
    base = _snap("2026-08-20", RANKS)
    cur = _snap("2026-09-17", RANKS)
    assert "ch" not in H._compare(cur, base)["AAA"]


def test_compare_omits_ch_when_churn_is_unavailable():
    base = _snap("2026-02-20", RANKS, metric_cols=[])
    cur = _snap("2026-09-17", RANKS, {"AAA": ["roic_pct"]})
    assert "ch" not in H._compare(cur, base)["AAA"]


def test_compare_leaves_new_entries_alone():
    base = _snap("2026-08-20", {"AAA": 1, "BBB": 2})
    cur = _snap("2026-09-17", RANKS)
    assert H._compare(cur, base)["CCC"] == {"new": True}


def test_build_history_carries_ch_into_the_payload(tmp_path):
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    universe = {f"T{i:03d}": i + 1 for i in range(60)}
    _df(universe).to_parquet(snaps / "2026-09-01_aaa.parquet")
    # Ranks must differ or the second run is excluded as a warm-start duplicate
    # - a real guard in `select_comparable_runs`, not an artifact of the test.
    moved = dict(universe)
    moved["T005"], moved["T006"] = universe["T006"], universe["T005"]
    _df(moved, {"T005": ["roic_pct", "peg_ratio_pct"]}).to_parquet(
        snaps / "2026-09-02_bbb.parquet")
    out = H.build_history(snapshots_dir=snaps)
    assert out["available"] is True
    assert out["delta"]["T005"]["prev"]["ch"] == [2, 0]
    assert "ch" not in out["delta"]["T006"]["prev"]


def test_build_history_on_real_snapshots_reports_some_churn():
    """Against the real directory the mechanism fires on a real, small minority
    of names - not zero (which would mean the wiring is dead) and not most of
    the universe (which would mean the schema rule is wrong)."""
    out = H.build_history()
    if not out.get("available"):
        pytest.skip("no comparable snapshot history available")
    entries = [e for d in out["delta"].values()
               for e in (d.get("m1"), d.get("prev")) if e]
    flagged = [e for e in entries if e.get("ch")]
    assert flagged, "no input churn found at all - the wiring is probably dead"
    assert len(flagged) < 0.5 * len(entries), "churn on most of the universe"


# ---------------------------------------------------------------------------
# The sentence - arming threshold
# ---------------------------------------------------------------------------

def test_arming_threshold_is_two():
    """Pinned as a number because the rest of the wording depends on it."""
    assert S.MIN_INPUT_CHURN == 2


def test_one_changed_metric_says_nothing():
    """Median |rank change| 7 vs 6 - indistinguishable from ordinary noise."""
    assert S._sentence_input_churn(_delta(1, 0), COMPARE) is None
    assert S._sentence_input_churn(_delta(0, 1), COMPARE) is None


def test_two_changed_metrics_fires():
    assert S._sentence_input_churn(_delta(2, 0), COMPARE) is not None


def test_one_lost_and_one_gained_fires():
    """Total churn is what was measured, not churn in either direction."""
    assert S._sentence_input_churn(_delta(1, 1), COMPARE) is not None


def test_no_churn_key_says_nothing():
    assert S._sentence_input_churn({"m1": {"dr": -40}}, COMPARE) is None


def test_no_history_says_nothing():
    assert S._sentence_input_churn(None, COMPARE) is None
    assert S._sentence_input_churn({}, COMPARE) is None


def test_new_stock_says_nothing():
    """A name with no baseline has no comparison to caveat."""
    assert S._sentence_input_churn({"m1": {"new": True}}, COMPARE) is None


@pytest.mark.parametrize("bad", [None, [], [3], [1, 2, 3], "2", 2, {"lost": 2}])
def test_malformed_ch_is_ignored_not_raised(bad):
    """A summary that raises blanks the whole drilldown for that stock."""
    assert S._sentence_input_churn({"m1": {"dr": -40, "ch": bad}}, COMPARE) is None


# ---------------------------------------------------------------------------
# The sentence - wording
# ---------------------------------------------------------------------------

def test_sentence_names_the_count_and_the_baseline():
    text = S._sentence_input_churn(_delta(3, 0), COMPARE)
    assert "3 of the metrics" in text
    assert "2026-08-20" in text and "28 days ago" in text


def test_sentence_distinguishes_lost_from_gained():
    lost = S._sentence_input_churn(_delta(2, 0), COMPARE)
    gained = S._sentence_input_churn(_delta(0, 2), COMPARE)
    assert "could be computed for" in lost and "cannot be now" in lost
    assert "could not be computed for" in gained and "can be now" in gained
    assert lost != gained


def test_sentence_reports_both_directions_when_both_occurred():
    text = S._sentence_input_churn(_delta(2, 1), COMPARE)
    assert "2 of the metrics" in text and "1 went the other way" in text


def test_sentence_explains_the_mechanism():
    """A caveat a reader cannot check is just a disclaimer. The sentence has to
    say *why* the score moved without the company moving."""
    text = S._sentence_input_churn(_delta(2, 0), COMPARE)
    assert "reweighted" in text
    assert "rather than a change in the company" in text


def test_sentence_carries_no_advice_language():
    for lost in range(0, 4):
        for gained in range(0, 4):
            if lost + gained < S.MIN_INPUT_CHURN:
                continue
            text = S._sentence_input_churn(_delta(lost, gained), COMPARE)
            assert text, (lost, gained)
            assert S.advice_terms_in(text) == [], (lost, gained, text)


def test_sentence_never_calls_churn_deterioration():
    """Churn leaves 52.4% of names worse off against a 44.6% base rate. It
    scatters ranks; it does not push them down. Wording it as bad news would
    invent a direction the measurement does not have."""
    text = S._sentence_input_churn(_delta(2, 1), COMPARE).lower()
    for word in ("deteriorat", "worse", "warning", "risk", "concern",
                 "weaken", "decline"):
        assert word not in text, f"'{word}' asserts a direction churn does not have"


def test_sentence_direction_is_not_taken_from_the_rank_move():
    """The same churn must read identically whether the stock rose or fell."""
    up = S._sentence_input_churn(_delta(2, 0, dr=90), COMPARE)
    down = S._sentence_input_churn(_delta(2, 0, dr=-90), COMPARE)
    assert up == down


# ---------------------------------------------------------------------------
# The sentence - baseline agreement with the other change sentences
# ---------------------------------------------------------------------------

def test_uses_the_same_baseline_as_the_change_sentences():
    """A caveat attached to a different window than the move it qualifies is
    worse than no caveat. Both read `_pick_comparison`, so a delta carrying
    churn only at `prev` must not caveat an `m1` move."""
    delta = {"m1": {"dr": -40}, "prev": {"dr": -3, "ch": [2, 0]}}
    assert S._pick_comparison(delta)[0] == "m1"
    assert S._sentence_input_churn(delta, COMPARE) is None


def test_falls_back_to_prev_when_that_is_the_chosen_baseline():
    delta = {"prev": {"dr": -3, "ch": [2, 0]}}
    text = S._sentence_input_churn(delta, COMPARE)
    assert text and "2026-09-16" in text


# ---------------------------------------------------------------------------
# build_summary integration
# ---------------------------------------------------------------------------

def _detail() -> dict:
    return {"ticker": "AAA", "rank": 40, "composite": 61.0,
            "cat_scores": {c: 50.0 for c in S.CATEGORIES},
            "contrib": {c: 6.0 for c in S.CATEGORIES},
            "pct": {}, "raw": {}, "peers": [], "flags": {}}


def _summary(delta):
    return S.build_summary(_detail(), universe_size=502, metric_meta={},
                           metric_weights={}, history_delta=delta,
                           history_compare=COMPARE)


def test_build_summary_includes_the_fact():
    kinds = [f["k"] for f in _summary(_delta(2, 0))]
    assert "input_churn" in kinds


def test_build_summary_omits_it_below_the_threshold():
    kinds = [f["k"] for f in _summary(_delta(1, 0))]
    assert "input_churn" not in kinds


def test_fact_follows_the_change_sentences_it_qualifies():
    """Order is the point: a reader must not meet the move without the caveat."""
    kinds = [f["k"] for f in _summary(_delta(2, 0))]
    assert kinds.index("input_churn") > kinds.index("change")
    assert kinds.index("input_churn") > kinds.index("change_driver")


def test_fact_precedes_the_general_confidence_sentence():
    """`confidence` states the coverage *level*; this states the *change*. The
    specific claim about this comparison belongs before the general one."""
    kinds = [f["k"] for f in _summary(_delta(2, 0))]
    if "confidence" in kinds:
        assert kinds.index("input_churn") < kinds.index("confidence")


def test_summary_text_round_trips_the_fact():
    text = S.summary_text(_summary(_delta(2, 0)))
    assert "changed availability" in text or "could be computed" in text
    assert S.advice_terms_in(text) == []
